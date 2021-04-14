%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Author: Shoichiro Takeda, Nippon Telegraph and Telephone Corporation
% Date (last update): 2021/04/14
% License: Please refer to the attached LICENCE file
%
% Please refer to the original paper: 
%   "Video Magnification in the Wild:", CVPR 2018
%
% All code provided here is to be used for **research purposes only**. 
%
% This implementation also includes some lightly-modified third party codes:
%   - matlabPyrTools, from "https://github.com/LabForComputationalVision/matlabPyrTools"
%   - myPyToolsExt&Filters from "http://people.csail.mit.edu/nwadhwa/phase-video/PhaseBasedRelease_20131023.zip"
%   - main & myfuntions,  from "https://github.com/acceleration-magnification/sources (*initial version)"
% All credit for the third party codes is with the authors.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
close all;
clear all;

% Add path
addpath(fullfile(pwd, 'outputs'));
addpath(fullfile(pwd, 'myPyrToolsExt'));
addpath(fullfile(pwd, 'myfunctions/Filters'));
addpath(fullfile(pwd, 'myfunctions/utilize'));
addpath(fullfile(pwd, 'Filters'));

% Set dir
dataDir = 'C:\Users\shoichirotakeda\Data\Video'; % Change your dir
% dataDir = '/Users/shoichirotakeda/Data/Video';
outputDir = [pwd, '\outputs'];

% Select input video
inFile = fullfile(dataDir,['ukulele.mp4']); % Change your data name
[Path,FileName,Ext] = fileparts(inFile);

% Read video
fprintf('Read input video\n');
vr = VideoReader(inFile);
vid = vr.read();

% Set video parameter
FrameRate = round(vr.FrameRate);
[nH, nW, nC, nF]= size(vid);
fprintf('Original VideoSize ... nH:%d, nW:%d, nC:%d, nF:%d\n', nH, nW, nC, nF);

% Set CSF parameter
nOri = 8; % number of orientations
nProp = 5; % fix all video in CVPR2018

% Set magnification parameter (gun.avi)
% ScaleVideoSize = 8/10;
% StartFrame = 1;
% EndFrame   = nF;
% alpha = 100;
% targetFreq = 20;
% fs = 480;
% beta = 0.5;
% FAF_weight = 1;

% Set magnification parameter (ukulele.mp4)
ScaleVideoSize = 1/3;
StartFrame = 4.8*FrameRate;
EndFrame   = StartFrame+10*FrameRate;
alpha = 260;
targetFreq = 40;
fs = 240;
beta = 1;
FAF_weight = 5;

% Set output Name
resultName = ['mag-vmw-',FileName, ...
            '-scale-' num2str(ScaleVideoSize) ...
            '-ori-' num2str(nOri) ...
            '-fs-' num2str(fs) ...
            '-ft-' num2str(targetFreq) ...
            '-alp-' num2str(alpha) ...
            '-beta-' num2str(beta) ...
            '-wFAF-' num2str(FAF_weight) ...
            ];
        
%% Preprocess for input video
% Resize (H x W) scale
TmpVid = imresizeVideo(vid, ScaleVideoSize);
clear vid;
vid = TmpVid;
clear TmpVid;

% Resize time
TmpVid = vid(:,:,:,StartFrame:EndFrame);
clear vid;
vid = TmpVid;

% Get final input video parameter
[nH, nW, nC, nF]= size(vid);
fprintf('Resized VideoSize ... nH:%d, nW:%d, nC:%d, nF:%d\n', nH, nW, nC, nF);

% Change RGB to YIQ color space & extract only Y color space
originalFrame = zeros(nH, nW, nC, nF, 'single');
for i = 1:1:nF 
    originalFrame(:, :, :, i) = single(rgb2ntsc(im2single(vid(:, :, :, i))));   
end

% Perform 2D FFT
fft_Y = single(fftshift(fftshift(fft2(squeeze(originalFrame(:,:,1,:))),1),2));

%% Get complex steerable filters (CSF) and filter indices (determine the filtering area in Fouried domain)
% Get maximum pyramid levels
ht = maxSCFpyrHt(zeros(nH,nW));

% Get CSF and indices
[CSF, filtIDX] = getCSFandIDX([nH nW], 2.^[0:-0.5:-ht], nOri, 'twidth', 0.75);

% Get pyramid patameter
nPyrLevel = size(CSF,1);

%% Get pyramid scale facter : lambda (Eq.6)
lambda =  zeros(nPyrLevel, nOri);
for level = 1:1:nPyrLevel
    if level == 1 || level == nPyrLevel
        tmp_h_down = size(CSF{level,1},1) ./ size(CSF{1,1},1);
        tmp_w_down = size(CSF{level,1},2) ./ size(CSF{1,1},2);
        DownSamplingFacter = (tmp_h_down + tmp_w_down) ./ 2;
        lambda(level,1) = 1/DownSamplingFacter;
    else
        for ori = 1:1:nOri
            tmp_h_down = size(CSF{level,ori},1) ./ size(CSF{1,1},1);
            tmp_w_down = size(CSF{level,ori},2) ./ size(CSF{1,1},2);
            DownSamplingFacter = (tmp_h_down + tmp_w_down) ./ 2;
            lambda(level,ori) = 1/DownSamplingFacter;
        end
    end
end

%% Calculate phase difference
fprintf('\n');
fprintf('Calculating Amplitude & Phase\n');

for level = 2:1:nPyrLevel-1 % except for the highest/lowest pyramid level
    for ori = 1:1:nOri
        fprintf('Processing pyramid level: %d, orientation: %d\n', level, ori);
    
        hIDX = filtIDX{level,ori}{1};
        wIDX = filtIDX{level,ori}{2};
        cfilter = CSF{level,ori};       

        for f = 1:nF
            % here, we apply rondomized sparcification algorhithm
            CSF_fft_Y = cfilter .* fft_Y(hIDX, wIDX, f);  
            R = ifft2(ifftshift(CSF_fft_Y)); 

            if f == 1
                phaseRef = angle(R);
                if license('test', 'Parallel Computing Toolbox') && canUseGPU()
                    phase = gpuArray( zeros(nF, numel(hIDX), numel(wIDX), 'single') );
                else
                    phase = zeros(nF, numel(hIDX), numel(wIDX), 'single');
                end
                norm_amp{level,ori} = zeros(nF, numel(hIDX), numel(wIDX), 'single');
            end

            ampCurrent = abs(R);
            phaseCurrent = angle(R);
            
            %%% Amplitude Normalization %%%
            norm_ampCurrenct = ampCurrent ./ lambda(level,ori);

            %%% whitening %%%
            N = numel(norm_ampCurrenct);
            x = norm_ampCurrenct;
            u = (1/N) * sum(sum(x));
            sigma = sqrt( (1/(N-1)) * sum(sum( (x - u).^2 )));
            norm_ampCurrenct = (x - u) / sigma; % (Eq.17)
            
            phase(f,:,:) = mod(pi+phaseCurrent-phaseRef,2*pi)-pi;
            norm_amp{level,ori}(f,:,:) = single(norm_ampCurrenct);
        end

        fprintf('Phase Unwrapping \n');
        phase = unwrap(phase);
        
        % Phase-based video motion processing in SIGGRAPH 2013
        % filtered_phase{level,ori} = gather(permute(FIRWindowBP(permute(phase, [2,3,1]), (targetFreq-1/2)/fs, (targetFreq+1/2)/fs), [3,1,2]));

        fprintf('Temporal Acceleration Filtering \n');
        filtered_phase{level,ori} = gather(applyTAF(phase, 1, targetFreq, fs));

        fprintf('Create Jerk-Aware Filter\n'); % (Eq.1-5)
        JAF{level,ori} = gather(getJAF(phase, 1, targetFreq, fs, lambda(level,ori).*beta));
    end
end

%% Propagation Correction for Jerk-Aware Filter (Eq.7)
fprintf('\n');
fprintf('Propagation Correction for Jerk-Aware Filter\n');  

for ori = 1:1:nOri
    fprintf('Processing orientation: %d of %d\n', ori, nOri);

    for level = 2:1:nPyrLevel-1
        pJAF = JAF{level,ori}; % initialized

        for prop = level+1:level+(nProp-1)
            if prop <= nPyrLevel-1
                pJAF = pJAF .* myimresize3(JAF{prop,ori}, pJAF); % multiple cascade
            end
        end

        JAF{level,ori} = pJAF;
    end
    
end

%% Create Hierarchical Edge-Aware Regularization
fprintf('\n');
fprintf('Create Hierarchical Edge-Aware Regularization\n'); 

for ori = 1:1:nOri
    for level = 2:1:nPyrLevel-1 % except for the highest/lowest pyramid level
        fprintf('Processing pyramid level: %d, orientation: %d\n', level, ori);
        tmp_norm_amp = norm_amp{level,ori};
        
        for prop = level-floor(nProp/2):level+floor(nProp/2)
            if prop ~= level && prop >= 1 && prop <= nPyrLevel-1
                tmp_norm_amp = max( tmp_norm_amp, myimresize3(norm_amp{prop,ori}, tmp_norm_amp) );
            end
        end
        
%         norm_amp{level,ori} = tmp_norm_amp; % miss @ cvpr2019
        sigma = 1/lambda(level,ori);
        if license('test', 'Parallel Computing Toolbox') && canUseGPU()
            tmp_norm_amp = gpuArray(tmp_norm_amp);
        end
        
        for f = 1:1:nF
            g_tmp_norm_amp = imgaussfilt(tmp_norm_amp(f,:,:), sigma);            
            tmp_norm_amp(f,:,:) = ( g_tmp_norm_amp - min(g_tmp_norm_amp(:)) ) ./ ( max(g_tmp_norm_amp(:)) - min(g_tmp_norm_amp(:))+eps );
        end
        
        g_norm_amp{level,ori} = gather(tmp_norm_amp);
    end
end
clear norm_amp; % for releasing memory

%% Create Fractioanl Anisotropic Filter
fprintf('\n');
fprintf('Create Fractioanl Anisotropic Filter\n'); 

if size(dir('calcFA.mexw64'),1) == 0
mex -I"C:\Users\shoichirotakeda\Programs\MATLAB\Video Magnification in the Wild (forShare)\Eigen" ...
        calcFA.cpp ...
        '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
else
    disp('Exist Builded Mex file')
end

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

twindowSize = windowSize;
swindowSize = 4;

for level = 2:1:nPyrLevel-1
    fprintf('Processing pyramid level: %d\n', level);
    
    [~, tmp_h, tmp_w] = size(filtered_phase{level,1});
    tmp_subtle_phase= zeros(nF, tmp_h, tmp_w, nOri, 'single');
    
    for ori = 1:1:nOri
        tmp_subtle_phase(:,:,:,ori) = myimresize3(JAF{level,ori}.*filtered_phase{level,ori}, filtered_phase{level,1});
    end 
    
    FA = calcFA(tmp_subtle_phase, twindowSize, swindowSize);
    
    sigma = 1/lambda(level,ori);
    
    if license('test', 'Parallel Computing Toolbox') && canUseGPU()
        FA = gpuArray(FA);
    end

    for frameIDX = 1:1:nF  
        if sum(FA(frameIDX,:,:),'all') ~= 0
            g_tmp_FA = imgaussfilt(FA(frameIDX,:,:), sigma);
            FA(frameIDX,:,:) = ( g_tmp_FA - min(g_tmp_FA(:)) ) ./ ( max(g_tmp_FA(:)) - min(g_tmp_FA(:))+eps );
        end
    end

    FAF{level} = gather(FA) .^ FAF_weight;
end

%% Magnification
fprintf('\n');
fprintf('Magnification \n');

fft_magY = zeros(nH, nW, nF, 'single');

for level = 2:1:nPyrLevel-1 % except for the highest/lowest pyramid level
    for ori = 1:1:nOri
        fprintf('Processing pyramid level: %d, orientation: %d\n', level, ori);
        
        hIDX = filtIDX{level,ori}{1};
        wIDX = filtIDX{level,ori}{2};
        cfilter = CSF{level,ori};
        
        % detP = filtered_phase{level,ori}; 
        % detP = JAF{level,ori} .* filtered_phase{level,ori};
        resize_FAF = myimresize3(FAF{level}, filtered_phase{level,ori});    
        detP = resize_FAF .* g_norm_amp{level,ori} .* JAF{level,ori} .* filtered_phase{level,ori};

        for f = 1:nF
            CSF_fft_Y = cfilter .* fft_Y(hIDX, wIDX, f);  
            R = ifft2(ifftshift(CSF_fft_Y)); 
            magR = R .* exp( 1i * (alpha * squeeze(detP(f,:,:))));
            fft_magR = fftshift(fft2(magR));
            fft_magY(hIDX, wIDX, f) = fft_magY(hIDX, wIDX, f) + (2 * cfilter .* fft_magR);
        end
    end

    clear detP
end 

% Add the lowest pyramid level
hIDX = filtIDX{nPyrLevel,1}{1};
wIDX = filtIDX{nPyrLevel,1}{2};
cfilter = CSF{nPyrLevel,1};  
for f = 1:nF
    fft_magY(hIDX, wIDX, f) = fft_magY(hIDX, wIDX, f) + (fft_Y(hIDX, wIDX, f) .* cfilter .^2 ); 
end

%% Rendering Video
fprintf('\n');
fprintf('Rendering Video\n');
tic;

outFrame = originalFrame; 
for f = 1:nF
    magY = real(ifft2(ifftshift(fft_magY(:,:,f))));
    outFrame(:, :, 1, f) = magY; 
    outFrame(:, :, :, f) = ntsc2rgb(outFrame(:,:,:,f));       
end

fprintf('\n');
fprintf('Output Video\n');
outName = fullfile(outputDir,'pre.avi');
vidOut = VideoWriter(outName, 'Uncompressed AVI');
vidOut.FrameRate = FrameRate;
open(vidOut) 

outFrame_final = im2uint8(outFrame);
                     
writeVideo(vidOut, outFrame_final);

close(vidOut);

% %% Compress output data size via ffmpeg
% ! ffmpeg -i ./outputs/pre.avi -c:v libx264 -preset veryslow -crf 1 -pix_fmt yuv420p ./outputs/output.mp4
% fprintf('\n');
% fprintf('rename\n'); movefile('./outputs/output.mp4',['./outputs/', resultName, '.mp4']);
% fprintf('delete prefile\n'); delete('./outputs/pre.avi');
% fprintf('Done\n');

figure('position', [9.80000000000000,599.400000000000,3034,934.400000000000]);
set(gcf,'Visible', 'off');
set(gcf,'color',[0 0 0])
colormap jet;
level = 2;
ori = 5;
map_caxis = [0,1];
phase_caxis = [-0.3,0.3];
% F(nF) = struct('cdata',[],'colormap',[]);
tmp_FAF = myimresize3(FAF{level}, filtered_phase{level,ori});    
for t = 1:1:nF
disp(t);
subplot(2,4,1);
imagesc(vid(:,:,:,t));
axis off;

subplot(2,4,2);
imagesc( squeeze( JAF{level,ori}(t,:,:)) );
caxis(map_caxis);
% title('JAF')
axis off;

subplot(2,4,3);
imagesc( squeeze(tmp_FAF(t,:,:)) );
% title('FAF')
caxis(map_caxis);
axis off;

subplot(2,4,4);
imagesc( squeeze(g_norm_amp{level,ori}(t,:,:)) );
% title('HEAR')
caxis(map_caxis*0.5);
axis off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2,4,4+1);
imagesc(squeeze(filtered_phase{level,ori}(t,:,:)));
caxis([-0.20,0.20]);   
% title('acc')
axis off;

subplot(2,4,4+2);
imagesc(squeeze(JAF{level,ori}(t,:,:) .* filtered_phase{level,ori}(t,:,:)));
caxis([-0.13, 0.13]);
% title('jerk')
axis off;

subplot(2,4,4+3);
imagesc(squeeze(tmp_FAF(t,:,:) .* JAF{level,ori}(t,:,:) .* filtered_phase{level,ori}(t,:,:)));
caxis([-0.048, 0.048]);  
% title('jerk with FAF')
axis off;

subplot(2,4,4+4);
imagesc(squeeze(g_norm_amp{level,ori}(t,:,:) .* tmp_FAF(t,:,:) .* JAF{level,ori}(t,:,:) .* filtered_phase{level,ori}(t,:,:)));
caxis([-0.007, 0.007]);   
% title('our')
axis off;

F(t) = getframe(gcf);
end

fprintf('\n');
fprintf('Output Video\n');
outName = fullfile(outputDir,'pre.avi');
vidOut = VideoWriter(outName, 'Uncompressed AVI');
vidOut.FrameRate = FrameRate; 
open(vidOut) 

writeVideo(vidOut, F);

disp('Finished')
close(vidOut);

%% Compress output data size via ffmpeg
! ffmpeg -i ./outputs/pre.avi -c:v libx264 -preset veryslow -crf 1 -pix_fmt yuv420p ./outputs/output.mp4
fprintf('\n');
fprintf('rename\n'); movefile('./outputs/output.mp4',['./outputs/', resultName, '.mp4']);
fprintf('delete prefile\n'); delete('./outputs/pre.avi');
fprintf('Done\n');
