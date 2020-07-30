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
inFile = fullfile(dataDir,['gun.avi']);
[Path,FileName,Ext] = fileparts(inFile);

% �f�����̎擾
fprintf('Reading Video Information\n');
vr = VideoReader(inFile);
FrameRate = round(vr.FrameRate);

% Reading video
vid = vr.read();
[nH, nW, nC, nF]= size(vid);
fprintf('Original VideoSize ... nH:%d, nW:%d, nC:%d, nF:%d\n', nH, nW, nC, nF);

% Set Video Parameter
ScaleVideoSize = 8/10;
StartFrame = 1;
EndFrame   = nF;

% Set Parameter
orientations = 8;

alpha1 = 7;
alpha2 = 16;
alpha3 = 100;
targetFreq = 20;
fs = 480;
beta = 0.5;

FAF_weight = 1;

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
        
%% �f���̓ǂݍ��݂Ɖ��H
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

%% �r�f�I��RGB����YIQ�ɒ�����Y���������o��
fprintf('Computing YIQ color Video\n');
originalFrame = zeros(nH, nW, nC, nF, 'single');
for i = 1:1:nF 
    originalFrame(:, :, :, i) = single(rgb2ntsc(im2single(vid(:, :, :, i))));   
end
% FFT2: �t�[���G�ϊ����f����(�t���[�����̉摜�ɑ΂���)������
% �f���ɑ΂��Ĉ�C��fft2����Ƃ��́Afftshift�̎d�l�ɋC��t����I
fprintf('Moving video to Fourier domain\n');
FFT_vid = single(fftshift(fftshift(fft2(squeeze(originalFrame(:,:,1,:))),1),2)); % ��������GPU�ɂ���ƁCR�Ƃ��̌v�Z���Ȃ����x���Ȃ�̂ł����cpu��

%% CSF(Complex Steerable Filter)�̐݌v
fprintf('Computing spatial filters\n');
ht = maxSCFpyrHt(zeros(nH,nW));

% 'halfOctave'����{�I�ɂ͍̗p
filters = getFilters([nH nW], 2.^[0:-0.5:-ht], orientations,'twidth', 0.75);
[croppedFilters, filtIDX] = getFilterIDX(filters);
fprintf('Using half octave bandwidth pyramid\n'); 

%% pyramid level����Scale Facter

nPyrLevels = numel(filters);  

DownSamplingFacter = zeros(nPyrLevels,1);
lambda =  zeros(nPyrLevels,1);
for i = 1:1:nPyrLevels-1
    tmp_h_down = size(croppedFilters{i},1) ./ size(croppedFilters{1},1);
    tmp_w_down = size(croppedFilters{i},2) ./ size(croppedFilters{1},2);
    DownSamplingFacter(i) = (tmp_h_down + tmp_w_down) ./ 2;
    lambda(i) = 1/DownSamplingFacter(i); % lambda�{���邱�ƂŁC���̉f���T�C�Y�Ɠ����ɂȂ�
end

%% Pyramid Index
pyrIDXs = zeros(orientations, (size(filters,2)-2)/orientations);
for i = 1:1:size(pyrIDXs,1) 
    for j = 1:1:size(pyrIDXs,2)
        pyrIDXs(i,j) = (i+1) + (j-1)*orientations;
    end
end

%% ���d�����ƈʑ����U���̌v�Z
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
        %%% z�ϊ��i���v�j���Ă���̂ŁC�s���~�b�h�̊K�w�ԂŔ�r�ł���
        N = numel(norm_ampCurrenct);
        x = norm_ampCurrenct;
        u = (1/N) * sum(sum(x));
        sigma = sqrt( (1/(N-1)) * sum(sum( (x - u).^2 )));
        norm_ampCurrenct = (x - u) / sigma; % (Eq.17)

        phaseDif(frameIDX,:,:) = mod(pi+phaseCurrent-phaseRef,2*pi)-pi; % �ʑ�����
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
        
        for w = oct-octave_width_half:oct+octave_width_half % ���ݒ��ڂ���octave�𒆐S�ɂƂ���octave_width���̏��ɒ���
            if w ~= oct && w >= 1 && w <= size(pyrIDXs,2)   % octave_width�͈͓̔��Œ�`��Ɏ��܂�octave�̏����C���ݒ��ڂ���octave�Ɏ������
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

if mod(windowSize,2) == 0 % windowSize�������̎� 
    x = linspace(-4*sigma, 4*sigma, windowSize+1);
else % windowSize����̎� 
    x = linspace(-4*sigma, 4*sigma, windowSize);
    windowSize = windowSize - 1;
end

% if size(dir('CreateAnisotropicSpectralFilter_expo.mexw64'),1) == 0
% mex -I"C:\Users\Shoichiro Takeda\Documents\NTT����\Amplitude-weighted Anisotoropic Spectral Masking for Video Magnification\Eigen" ...
%         Create_ADC_FA.cpp ...
%         '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
% else
%     disp('Exist Builded Mex file')
% end


twindowSize = windowSize; % ����S�{���ƏI���Ȃ��i1���Ԃ���̔�����*12�ŏI���j
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
    
    % 2D gaussian DAF
    sigma = DownSamplingFacter(pyrIDXs(ori,oct));  
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

%% ����
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
            tmp_DAF = original_imresize3_gpu(FAF{oct}, ac_delta_phase{level});    
            detP{3} = tmp_DAF .* g_norm_amp{level} .* JAF{level} .* ac_delta_phase{level};
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

%% output Video
inFile = fullfile(outputDir,['Learning_gun_dynamic5.mp4']);
[Path,FileName,Ext] = fileparts(inFile);
vr = VideoReader(inFile);
learning_vid = vr.read();
learning_vid = resizeVideo(learning_vid, ScaleVideoSize);

fprintf('\n');
fprintf('Output Video\n');
outName = fullfile(outputDir,[resultName, '.avi']);
vidOut = VideoWriter(outName, 'Uncompressed AVI');
vidOut.FrameRate = FrameRate; % ���f����Fr
% vidOut.Quality = 100;
open(vidOut) 

outFrame_final = vertcat(horzcat( vid, zeros(nH, 10, 3, nF), im2uint8(outFrame{3}) ), ...
                         zeros(10, nW*2 + 10, 3, nF) , ...
                         horzcat( learning_vid, zeros(nH, 10, 3, nF), im2uint8(outFrame{2}) ) ...
                         );

writeVideo(vidOut, outFrame_final);

disp('Finished')
close(vidOut);

% !ffmpeg -i wood.mp4 -c:v libx264 -preset veryslow -crf 0 out.mp4

toc;

pause;

%% camera ready

% file select
inFile = fullfile(outputDir,['Learning_gun_dynamic5.mp4']);
[Path,FileName,Ext] = fileparts(inFile);
vr = VideoReader(inFile);
learning_vid = vr.read();
learning_vid = resizeVideo(learning_vid, ScaleVideoSize);

% ROI_h = 210:330;      
% ROI_w = 330*ones(1,numel(ROI_h));
ROI_h = 170:260;  
ROI_w = 490*ones(1,numel(ROI_h));

imstack_original = zeros(size(ROI_w,2), size(vid,4), 3,'single');
imstack_acc = zeros(size(ROI_w,2), size(vid,4), 3,'single');
imstack_jerk = zeros(size(ROI_w,2), size(vid,4), 3,'single');
imstack_proposed = zeros(size(ROI_w,2), size(vid,4), 3,'single');
imstack_learning = zeros(size(ROI_w,2), size(vid,4), 3,'single');

for i = 1:1:size(vid,4)
    for j = 1:1:size(ROI_w,2)
        imstack_original(j,i,:) = squeeze(im2single(vid(ROI_h(j),ROI_w(j),:,i)));
%         imstack_acc(j,i,:)      = squeeze(outFrame{1}(ROI_h(j),ROI_w(j),:,i));
%         imstack_jerk(j,i,:)     = squeeze(outFrame{2}(ROI_h(j),ROI_w(j),:,i));
%         imstack_proposed(j,i,:) = squeeze(outFrame{3}(ROI_h(j),ROI_w(j),:,i));
        imstack_learning(j,i,:) = squeeze(im2single(learning_vid(ROI_h(j),ROI_w(j),:,i)));
    end
end

figure
ylims = [1,nF];
subplot(1,5,1);
imshow(permute((imstack_original(:,1:nF,:)), [1,2,3]))
% ylim(ylims);
subplot(1,5,2);
imshow(permute((imstack_acc(:,1:nF,:)), [1,2,3]))
% ylim(ylims);
subplot(1,5,3);
imshow(permute((imstack_jerk(:,1:nF,:)), [1,2,3]))
% ylim(ylims);
subplot(1,5,4);
imshow(permute((imstack_proposed(:,1:nF,:)), [1,2,3]))
% ylim(ylims);
subplot(1,5,5);
imshow(permute((imstack_learning(:,1:nF,:)), [1,2,3]))

ROI_w3 = 360:460; 
ROI_h3 = 361*ones(1,numel(ROI_w3));

imstack_original = zeros(size(ROI_w3,2), size(vid,4), 3,'single');
imstack_acc = zeros(size(ROI_w3,2), size(vid,4), 3,'single');
imstack_jerk = zeros(size(ROI_w3,2), size(vid,4), 3,'single');
imstack_proposed = zeros(size(ROI_w3,2), size(vid,4), 3,'single');
imstack_learning = zeros(size(ROI_w3,2), size(vid,4), 3,'single');

for i = 1:1:size(vid,4)
    for j = 1:1:size(ROI_w3,2)
        imstack_original(j,i,:) = squeeze(im2single(vid(ROI_h3(j),ROI_w3(j),:,i)));
        imstack_acc(j,i,:)      = squeeze(outFrame{1}(ROI_h3(j),ROI_w3(j),:,i));
        imstack_jerk(j,i,:)     = squeeze(outFrame{2}(ROI_h3(j),ROI_w3(j),:,i));
        imstack_proposed(j,i,:) = squeeze(outFrame{3}(ROI_h3(j),ROI_w3(j),:,i));
        imstack_learning(j,i,:) = squeeze(im2single(learning_vid(ROI_h3(j),ROI_w3(j),:,i)));
    end
end

figure
xlims = [20,50];
ylims = [40,95];
subplot(1,5,1);
imshow(permute((imstack_original(:,1:nF,:)), [1,2,3]))
subplot(1,5,2);
imshow(permute((imstack_acc(:,1:nF,:)), [1,2,3]))
subplot(1,5,3);
imshow(permute((imstack_jerk(:,1:nF,:)), [1,2,3]))
subplot(1,5,4);
imshow(permute((imstack_proposed(:,1:nF,:)), [1,2,3]))
subplot(1,5,5);
imshow(permute((imstack_learning(:,1:nF,:)), [1,2,3]))

