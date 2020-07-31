%% Author: Shoichiro Takeda
%% Date: 2018/10/17
clc;
close all;
clear all;
setPath;

% Select Video
dataDir = [pwd, '\data'];
outputDir = [pwd, '\results'];

% file select
inFile = fullfile(dataDir,['ukulele.mp4']);
[Path,FileName,Ext] = fileparts(inFile);

% 映像情報の取得
fprintf('Reading Video Information\n');
vr = VideoReader(inFile);
FrameRate = round(vr.FrameRate);

% Reading video
vid = vr.read();
[nH, nW, nC, nF]= size(vid);
fprintf('Original VideoSize ... nH:%d, nW:%d, nC:%d, nF:%d\n', nH, nW, nC, nF);

% Set Video Parameter
ScaleVideoSize = 1/3;
StartFrame = 4.8*FrameRate;
EndFrame   = StartFrame+10*FrameRate;

% Set Parameter
orientations = 8;

alpha1 = 20;
alpha2 = 25;
alpha3 = 260;
targetFreq = 40;
fs = 240;
beta = 1;

FAF_weight = 5;

%% Set output Name
magFactor = [alpha1, alpha2, alpha3]; 
resultName = ['Figure_',FileName, ...
            '-scale-' num2str(ScaleVideoSize) ...
            '-ori-' num2str(orientations) ...
            '-fs-' num2str(fs) ...
            '-ft-' num2str(targetFreq) ...
            '-alp-' num2str(magFactor) ...
            '-beta-' num2str(beta) ...
            '-wFAF-' num2str(FAF_weight) ...
            ];
        
%% 映像の読み込みと加工
% Resizing video
TmpVid = resizeVideo(vid, ScaleVideoSize);
clear vid;
vid = TmpVid;
clear TmpVid;

% Resizing Time
TmpVid = vid(:,:,:,StartFrame:EndFrame);
clear vid;
vid = TmpVid;

% Final Video Info
[nH, nW, nC, nF]= size(vid);
fprintf('Resized VideoSize ... nH:%d, nW:%d, nC:%d, nF:%d\n', nH, nW, nC, nF);

%% ビデオをRGBからYIQに直してYだけ抜き出す
fprintf('Computing YIQ color Video\n');
originalFrame = zeros(nH, nW, nC, nF, 'single');
for i = 1:1:nF 
    originalFrame(:, :, :, i) = single(rgb2ntsc(im2single(vid(:, :, :, i))));   
end
% FFT2: フーリエ変換を映像に(フレーム毎の画像に対して)かける
% 映像に対して一気にfft2するときは、fftshiftの仕様に気を付ける！
fprintf('Moving video to Fourier domain\n');
FFT_vid = single(fftshift(fftshift(fft2(squeeze(originalFrame(:,:,1,:))),1),2)); % こっからGPUにすると，Rとかの計算がなぜか遅くなるのでこれはcpuで

%% CSF(Complex Steerable Filter)の設計
fprintf('Computing spatial filters\n');
ht = maxSCFpyrHt(zeros(nH,nW));

% 'halfOctave'を基本的には採用
filters = getFilters([nH nW], 2.^[0:-0.5:-ht], orientations,'twidth', 0.75);
[croppedFilters, filtIDX] = getFilterIDX(filters);
fprintf('Using half octave bandwidth pyramid\n'); 

%% pyramid level毎のScale Facter

nPyrLevels = numel(filters);  

DownSamplingFacter = zeros(nPyrLevels,1);
lambda =  zeros(nPyrLevels,1);
for i = 1:1:nPyrLevels-1
    tmp_h_down = size(croppedFilters{i},1) ./ size(croppedFilters{1},1);
    tmp_w_down = size(croppedFilters{i},2) ./ size(croppedFilters{1},2);
    DownSamplingFacter(i) = (tmp_h_down + tmp_w_down) ./ 2;
    lambda(i) = 1/DownSamplingFacter(i); % lambda倍することで，元の映像サイズと同じになる
end

%% Pyramid Index
pyrIDXs = zeros(orientations, (size(filters,2)-2)/orientations);
for i = 1:1:size(pyrIDXs,1) 
    for j = 1:1:size(pyrIDXs,2)
        pyrIDXs(i,j) = (i+1) + (j-1)*orientations;
    end
end

%% 多重分解と位相＆振幅の計算
fprintf('\n');
fprintf('Calculating Amplitude & Phase\n');
tic;
for level = 2:nPyrLevels
    fprintf('Processing Pyramid Level %d of %d\n', level, nPyrLevels);
    
    hIDXs = filtIDX{level, 1};
    wIDXs = filtIDX{level, 2};
    cfilter = croppedFilters{level};       
    
    for frameIDX = 1:nF              
        CSF_FFT_vid = cfilter .* FFT_vid(hIDXs, wIDXs, frameIDX);  
        R = ifft2(ifftshift(CSF_FFT_vid)); 
                
        if frameIDX == 1
            phaseRef = angle(R);    
            phaseDif = gpuArray( zeros(nF, numel(hIDXs), numel(wIDXs), 'single') );
            ampLevel = gpuArray( zeros(nF, numel(hIDXs), numel(wIDXs), 'single') );
        end
        
        ampCurrent = abs(R);
        phaseCurrent = angle(R);
        
        %%% Amplitude Normalization %%%
        norm_ampCurrenct = ampCurrent ./ lambda(level);
        
        %%% whitening %%%
        %%% z変換（統計）しているので，ピラミッドの階層間で比較できる
        N = numel(norm_ampCurrenct);
        x = norm_ampCurrenct;
        u = (1/N) * sum(sum(x));
        sigma = sqrt( (1/(N-1)) * sum(sum( (x - u).^2 )));
        norm_ampCurrenct = (x - u) / sigma; % (Eq.17)

        phaseDif(frameIDX,:,:) = mod(pi+phaseCurrent-phaseRef,2*pi)-pi; % 位相差分
        ampLevel(frameIDX,:,:) = norm_ampCurrenct; 
    end
    
    fprintf('Phase Unwrapping \n');
    delta_phase = unwrap(phaseDif);
    norm_amp{level} = gather(single(ampLevel));
    
    if 1 < level && level ~= nPyrLevels
        fprintf('Temporal Acceleration Filtering \n');
        ac_delta_phase{level} = gather(tempkernel_acceleration(delta_phase, targetFreq, fs));
          
        fprintf('Create Jerk-Aware Filter\n');        
        JAF{level} = gather(CreateJerkAwareFilter(delta_phase, targetFreq, fs, lambda(level).*beta));
    end
end
toc;

%% Propagation Correction for Jerk-Aware Filter
fprintf('\n');
fprintf('Propagation Correction for Jerk-Aware Filter\n');  
tic;
nProp = 5;
for ori = 1:1:size(pyrIDXs,1)
    fprintf('Processing Orientation %d of %d\n', ori, size(pyrIDXs,1));

    for oct = 1:1:size(pyrIDXs,2)
        pJAF = JAF{pyrIDXs(ori,oct)}; % initialized

        for octProp = oct+1:oct+(nProp-1)
            if octProp <= size(pyrIDXs,2)
                pJAF = pJAF .* original_imresize3_gpu(JAF{pyrIDXs(ori,octProp)}, pJAF); % multiple cascade
            end
        end

        JAF{pyrIDXs(ori,oct)} = pJAF;
    end
end
toc;

% %% Subtle Phase Calculation (Jerk-Aware Video Acceleration Magnification)
% for level = 2:nPyrLevels-1
%     subtle_phase{level} = JAF{level} .* ac_delta_phase{level};
% end
% clear ac_delta_phase
% clear JAF

%% Create Hierarchinal Amplitude Mask
fprintf('\n');
fprintf('Create Hierarchinal Amplitude Mask\n'); 
tic;
octave_width = 5; % pyramid level width of interest
octave_width_half = ceil(octave_width / 2)-1;

for ori = 1:1:size(pyrIDXs,1) 
    for oct = 1:1:size(pyrIDXs,2)
        fprintf('Processing Pyramid Level %d of %d\n', pyrIDXs(ori,oct), nPyrLevels);
        
        tmp_norm_amp = norm_amp{pyrIDXs(ori,oct)};
        
        for w = oct-octave_width_half:oct+octave_width_half % 現在注目するoctaveを中心にとしたoctave_width内の情報に注目
            if w ~= oct && w >= 1 && w <= size(pyrIDXs,2)   % octave_widthの範囲内で定義域に収まるoctaveの情報を，現在注目するoctaveに取り入れる
                res_norm_amp_w = original_imresize3_gpu(norm_amp{pyrIDXs(ori,w)}, tmp_norm_amp);
                tmp_norm_amp = max(tmp_norm_amp, res_norm_amp_w);
            end
        end
        
        norm_amp{pyrIDXs(ori,oct)} = tmp_norm_amp;
    end
end
toc;

fprintf('\n');
fprintf('2D Gaussian Smoothing\n'); 
tic;
g_norm_amp = norm_amp; %% initialized
for ori = 1:1:size(pyrIDXs,1) 
    for oct = 1:1:size(pyrIDXs,2)
        fprintf('Processing Pyramid Level %d of %d\n', pyrIDXs(ori,oct), nPyrLevels);

        sigma = DownSamplingFacter(pyrIDXs(ori,oct));
        tmp_norm_amp = gpuArray(norm_amp{pyrIDXs(ori,oct)});
        
        for frameIDX = 1:1:nF  
            g_tmp_norm_amp = imgaussfilt(tmp_norm_amp(frameIDX,:,:), sigma);            
            tmp_norm_amp(frameIDX,:,:) = ( g_tmp_norm_amp - min(g_tmp_norm_amp(:)) ) ./ ( max(g_tmp_norm_amp(:)) - min(g_tmp_norm_amp(:))+eps );
        end
        
        g_norm_amp{pyrIDXs(ori,oct)} = gather(tmp_norm_amp);  
    end
end
toc;

%% Diffusion Aware Filter
fprintf('\n');
fprintf('Create ADC & FA\n');

windowSize = ceil(fs / (4 * targetFreq)); 
sigma      = windowSize/sqrt(2);

if windowSize < 3
    windowSize = 3;
end

if mod(windowSize,2) == 0 % windowSizeが偶数の時 
    x = linspace(-4*sigma, 4*sigma, windowSize+1);
else % windowSizeが奇数の時 
    x = linspace(-4*sigma, 4*sigma, windowSize);
    windowSize = windowSize - 1;
end

% if size(dir('CreateAnisotropicSpectralFilter_expo.mexw64'),1) == 0
% mex -I"C:\Users\Shoichiro Takeda\Documents\NTT研究\Amplitude-weighted Anisotoropic Spectral Masking for Video Magnification\Eigen" ...
%         Create_ADC_FA.cpp ...
%         '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
% else
%     disp('Exist Builded Mex file')
% end

twindowSize = windowSize; % これ４倍だと終わらない（1時間からの半減期*12で終わる）
swindowSize = 4;

fprintf('Calculating Start\n');
for level = 2:orientations:nPyrLevels-1
    tic;
    [ori,oct] = find(pyrIDXs==level);
    fprintf('Calculating ADC&FA @ Octave Level %d / %d\n', oct, size(pyrIDXs,2));
    
    [~, tmp_h, tmp_w] = size(ac_delta_phase{level});
    tmp_subtle_phase= zeros(nF, tmp_h, tmp_w, orientations, 'single');
    
    for ori = 1:1:orientations
        tmp_subtle_phase(:,:,:,ori) = original_imresize3_gpu(JAF{level+(ori-1)} .* ac_delta_phase{level+(ori-1)}, ac_delta_phase{level});
    end 
    
    [~, FA{oct}] = Create_ADC_FA(tmp_subtle_phase, twindowSize, swindowSize);
    
    toc;
end

fprintf('\n');
fprintf('Create Diffusion Aware Filter\n');
fprintf('Calculating Start\n');
for oct = 1:1:size(pyrIDXs,2)
    tic;
    
    % 2D gaussian FAF
    sigma =  DownSamplingFacter(pyrIDXs(ori,oct));  
    tmp_FA = gpuArray(FA{oct});

    for frameIDX = 1:1:nF  
        if ~isequal(tmp_FA(frameIDX,:,:), zeros(size(tmp_FA(frameIDX,:,:))))
            g_tmp_FA = imgaussfilt(tmp_FA(frameIDX,:,:), sigma);
            tmp_FA(frameIDX,:,:) = ( g_tmp_FA - min(g_tmp_FA(:)) ) ./ ( max(g_tmp_FA(:)) - min(g_tmp_FA(:))+eps );
        end
    end
    
    FAF{oct} = gather(tmp_FA) .^ FAF_weight;

    toc;
end

%% 強調
fprintf('\n');
fprintf('Magnification');

for magIDX = 1:1:size(magFactor,2)
    fft_magY{magIDX} = zeros(nH, nW, nF, 'single');
end

for level = 2:nPyrLevels
    fprintf('Processing Pyramid Level %d of %d\n', level, nPyrLevels);
    
    hIDXs = filtIDX{level, 1};
    wIDXs = filtIDX{level, 2};
    cfilter = croppedFilters{level};      
    
    if 1 < level
        if level ~= nPyrLevels
            fprintf('VAM\n');
            detP{1} = ac_delta_phase{level};
         
            fprintf('JAVAM\n'); 
            detP{2} = JAF{level} .* ac_delta_phase{level};

%             fprintf('HAmp JAVAM\n');
%             detP{2} = g_norm_amp{level} .* subtle_phase{level};
            
            fprintf('DAF HAmp JAVAM\n');
            [~,oct] = find(pyrIDXs==level);
            tmp_FAF = original_imresize3_gpu(FAF{oct}, ac_delta_phase{level});    
            detP{3} = tmp_FAF .* g_norm_amp{level} .* JAF{level} .* ac_delta_phase{level};
        end
        
        fprintf('Amplifying \n');
        for i = 1:1:size(magFactor,2)
            for frameIDX = 1:nF
                CSF_FFT_vid = cfilter .* FFT_vid(hIDXs, wIDXs, frameIDX);  
                R = ifft2(ifftshift(CSF_FFT_vid)); 
                if level == numel(filters)
                    fft_magY{i}(hIDXs, wIDXs, frameIDX) = fft_magY{i}(hIDXs, wIDXs, frameIDX) + (FFT_vid(hIDXs, wIDXs, frameIDX) .* cfilter .^2 );   
                else
                    magR = R .* exp( 1i * (magFactor(i) * squeeze(detP{i}(frameIDX,:,:))));
                    fft_magR = fftshift(fft2(magR));
                    fft_magY{i}(hIDXs, wIDXs, frameIDX) = fft_magY{i}(hIDXs, wIDXs, frameIDX) + (2 * cfilter .* fft_magR);
                end
            end
        end
    end
        
    clear detP
end 

%% Rendering Video
fprintf('\n');
fprintf('Rendering Video\n');
tic;
for magIDX = 1:1:size(magFactor,2)
    outFrame{magIDX} = originalFrame; 
end

for i = 1:1:size(magFactor,2)
    fprintf('Processing Video %d of %d\n', i, size(magFactor,2));
    for frameIDX = 1:nF
        magY = real(ifft2(ifftshift(fft_magY{i}(:,:,frameIDX))));
        outFrame{i}(:, :, 1, frameIDX) = magY; 
        outFrame{i}(:, :, :, frameIDX) = ntsc2rgb(outFrame{i}(:,:,:,frameIDX));       
    end
end

% %% figure 2
% figure('position', [9.80000000000000,599.400000000000,3035.20000000000,934.400000000000]);
% colormap jet;
% level = 6 + 8*0;
% [~,oct] = find(pyrIDXs==level);
% map_caxis = [0,1];
% phase_caxis = [-0.3,0.3];
% t = 244;
% subplot(2,5,1);
% imagesc(vid(:,:,:,t));
% axis off;
% 
% subplot(2,5,2);
% imagesc( squeeze( JAF{level}(t,:,:)) );
% caxis(map_caxis);
% title('JAF')
% axis off;
% 
% subplot(2,5,3);
% [~,oct] = find(pyrIDXs==level);
% tmp_FAF = original_imresize3_gpu(FAF{oct}, ac_delta_phase{level});    
% imagesc( squeeze(tmp_FAF(t,:,:)) );
% title('FAF')
% caxis(map_caxis);
% axis off;
% 
% subplot(2,5,4);
% imagesc( squeeze(g_norm_amp{level}(t,:,:)) );
% title('HEAR')
% caxis(map_caxis*0.5);
% axis off;
% 
% subplot(2,5,6);
% imagesc(squeeze(ac_delta_phase{level}(t,:,:)));
% caxis([-0.20,0.20]);  
% title('jerk')
% axis off;
% 
% subplot(2,5,7);
% imagesc(squeeze(JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.13, 0.13]);  
% title('jerk')
% axis off;
% 
% subplot(2,5,8);
% imagesc(squeeze(tmp_FAF(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.048, 0.048]);  
% title('jerk with FAF')
% axis off;
% 
% subplot(2,5,9);
% imagesc(squeeze(g_norm_amp{level}(t,:,:) .* tmp_FAF(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.007, 0.007]);  
% title('our')
% axis off;
% 
% subplot(2,5,10);
% imagesc(squeeze(g_norm_amp{level}(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.017, 0.017]); 
% title('jerk with HEAR')
% axis off;

%% Movie 2
% clear F
% % figure('position', [9.80000000000000,599.400000000000,3034,934.400000000000]);
% figure('position', [230.600000000000,761,2724.80000000000,683.200000000000]);
% set(gcf,'Visible', 'off');
% set(gcf,'color',[0 0 0])
% colormap jet;
% level = 6 + 8*0;
% [~,oct] = find(pyrIDXs==level);
% map_caxis = [0,1];
% phase_caxis = [-0.3,0.3];
% % F(nF) = struct('cdata',[],'colormap',[]);
% tmp_FAF = original_imresize3_gpu(FAF{oct}, ac_delta_phase{level});    
% for t = 1:1:nF
% disp(t);
% subplot(2,5,1);
% imagesc(vid(:,:,:,t));
% axis off;
% 
% subplot(2,5,2);
% imagesc( squeeze( JAF{level}(t,:,:)) );
% caxis(map_caxis);
% % title('JAF')
% axis off;
% 
% subplot(2,5,3);
% [~,oct] = find(pyrIDXs==level);
% tmp_FAF = original_imresize3_gpu(FAF{oct}, ac_delta_phase{level});    
% imagesc( squeeze(tmp_FAF(t,:,:)) );
% % title('FAF')
% caxis(map_caxis);
% axis off;
% 
% subplot(2,5,4);
% imagesc( squeeze(g_norm_amp{level}(t,:,:)) );
% % title('HEAR')
% caxis(map_caxis*0.5);
% axis off;
% 
% subplot(2,5,6);
% imagesc(squeeze(ac_delta_phase{level}(t,:,:)));
% caxis([-0.20,0.20]);  
% % title('jerk')
% axis off;
% 
% subplot(2,5,7);
% imagesc(squeeze(JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.13, 0.13]);  
% % title('jerk')
% axis off;
% 
% subplot(2,5,8);
% imagesc(squeeze(tmp_FAF(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.048, 0.048]);  
% % title('jerk with FAF')
% axis off;
% 
% subplot(2,5,9);
% imagesc(squeeze(g_norm_amp{level}(t,:,:) .* tmp_FAF(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.007, 0.007]);  
% % title('our')
% axis off;
% 
% subplot(2,5,10);
% imagesc(squeeze(g_norm_amp{level}(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
% caxis([-0.017, 0.017]); 
% % title('jerk with HEAR')
% axis off;
% 
% F(t) = getframe(gcf);
% end
% 
% fprintf('\n');
% fprintf('Output Video\n');
% outName = fullfile(outputDir,'MotionMagnificationMethod_cameraready.avi');
% vidOut = VideoWriter(outName, 'Uncompressed AVI');
% vidOut.FrameRate = FrameRate; % 元映像のFr
% open(vidOut) 
% 
% writeVideo(vidOut, F);
% 
% disp('Finished')
% close(vidOut);

%% 

% figure('position', [9.80000000000000,599.400000000000,3034,934.400000000000]);
level = 6 + 8*0;
[~,oct] = find(pyrIDXs==level);
map_caxis = [0,1];
phase_caxis = [-0.3,0.3];
% F(nF) = struct('cdata',[],'colormap',[]);
tmp_FAF = original_imresize3_gpu(FAF{oct}, ac_delta_phase{level});    

for i = 10
disp(i);
clear F

for t = 1:1:nF
figure('position',[1256.20000000000,1392.20000000000,403.200000000000,233.600000000000]);
set(gcf,'Visible', 'off');
set(gcf,'color',[0 0 0])
colormap jet;
disp(t);

if i == 1
    imagesc(vid(:,:,:,t));
    axis off;
    
elseif i==2
    imagesc( squeeze( JAF{level}(t,:,:)) );
    caxis(map_caxis);
    axis off;
    
elseif i==3
    [~,oct] = find(pyrIDXs==level);
    tmp_FAF = original_imresize3_gpu(FAF{oct}, ac_delta_phase{level});    
    imagesc( squeeze(tmp_FAF(t,:,:)) );
    caxis(map_caxis);
    axis off;
    
elseif i == 4
    imagesc( squeeze(g_norm_amp{level}(t,:,:)) );
    caxis(map_caxis*0.5);
    axis off;
    
elseif i==6
    imagesc(squeeze(ac_delta_phase{level}(t,:,:)));
    caxis([-0.20,0.20]);  
    axis off;

elseif i==7
    imagesc(squeeze(JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
    caxis([-0.13, 0.13]);
    axis off;
    
elseif i==8
    imagesc(squeeze(tmp_FAF(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
    caxis([-0.048, 0.048]);  
    axis off;
    
elseif i==9
    imagesc(squeeze(g_norm_amp{level}(t,:,:) .* tmp_FAF(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
    caxis([-0.007, 0.007]);  
    axis off;
    
elseif i == 10
    imagesc(squeeze(g_norm_amp{level}(t,:,:) .* JAF{level}(t,:,:) .* ac_delta_phase{level}(t,:,:)));
    caxis([-0.017, 0.017]); 
    axis off;
    
end

F(t) = getframe(gcf);
end

fprintf('\n');
fprintf('Output Video\n');
outName = fullfile(outputDir,['MotionMagnificationMethod_cameraready',num2str(i),'.avi']);
vidOut = VideoWriter(outName, 'Uncompressed AVI');
vidOut.FrameRate = FrameRate; % 元映像のFr
open(vidOut) 

writeVideo(vidOut, F);

disp('Finished')
close(vidOut);

close all

end