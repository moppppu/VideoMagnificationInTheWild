%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Author: Shoichiro Takeda, Nippon Telegraph and Telephone Corporation
% Date (last update): 2021/04/14
% License: Please refer to the attached LICENCE file
%
% Please refer to the original paper: 
%   "﻿Video Magnification in the Wild Using Fractional Anisotropy in
%   Temporal Distribution", CVPR 2019
%
% All code provided here is to be used for **research purposes only**. 
%
% This implementation also includes some lightly-modified third party codes:
%   - myPyToolsExt, from "http://people.csail.mit.edu/nwadhwa/phase-video/PhaseBasedRelease_20131023.zip"
%   - main_vmw & myfuntions,  from "https://github.com/acceleration-magnification/sources (*initial version)"
%   - Eigen, from "http://eigen.tuxfamily.org/index.php?title=Main_Page"
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
% dataDir = 'C:\Users\shoichirotakeda\Data\Video'; % Change your dir
dataDir = '/Users/shoichirotakeda/Data/Video';
outputDir = [pwd];

% Select input video
inFile = fullfile(dataDir,['gun.mp4']); % Change your data name
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
nProp = 5; % fix all video in "Jerk-Aware Video Acceleration Magnification" @ CVPR2018

% Set magnification parameter (gun.avi)
ScaleVideoSize = 8/10;
StartFrame = 1;
EndFrame   = nF;
alpha = 100;
targetFreq = 20;
fs = 480;
beta = 0.5;
FAF_weight = 1;

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

%% Get complex steerable filters (CSF) and filter indices (determine the filtering area in Fourier domain)
% Get maximum pyramid levels
ht = maxSCFpyrHt(zeros(nH,nW));

% Get CSF and indices
[CSF, filtIDX] = getCSFandIDX([nH nW], 2.^[0:-0.5:-ht], nOri, 'twidth', 0.75);

% Get pyramid patameter
nPyrLevel = size(CSF,1);

%% Get pyramid scale facter : lambda
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

            %%% Whitening %%%
            N = numel(norm_ampCurrenct);
            x = norm_ampCurrenct;
            u = (1/N) * sum(sum(x));
            sigma = sqrt( (1/(N-1)) * sum(sum( (x - u).^2 )));
            norm_ampCurrenct = (x - u) / sigma;
            
            phase(f,:,:) = mod(pi+phaseCurrent-phaseRef,2*pi)-pi;
            norm_amp{level,ori}(f,:,:) = single(norm_ampCurrenct);
        end

        fprintf('Phase Unwrapping \n');
        phase = unwrap(phase);
        
        fprintf('Temporal Acceleration Filtering \n');
        filtered_phase{level,ori} = gather(applyTAF(phase, 1, targetFreq, fs));

        fprintf('Create Jerk-Aware Filter\n');
        JAF{level,ori} = gather(getJAF(phase, 1, targetFreq, fs, lambda(level,ori).*beta));
    end
end

%% Propagation Correction for Jerk-Aware Filter
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
    mex -I"C:\hoge\Eigen" calcFA.cpp '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"' % Change your Eigen dir
else
    disp('Exist Builded Mex file');
end

twindowSize = ceil(fs / (4 * targetFreq)); 

if twindowSize < 3
    twindowSize = 3;
end

if mod(twindowSize,2) ~= 0
    twindowSize = twindowSize - 1;
end

swindowSize = 4; % We fixed all expetiments

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

outFrame = originalFrame; 
for f = 1:nF
    magY = real(ifft2(ifftshift(fft_magY(:,:,f))));
    outFrame(:, :, 1, f) = magY; 
    outFrame(:, :, :, f) = ntsc2rgb(outFrame(:,:,:,f));       
end

fprintf('\n');
fprintf('Output Video\n');
outName = fullfile(outputDir,'output.avi');
vidOut = VideoWriter(outName, 'Uncompressed AVI');
vidOut.FrameRate = FrameRate;
open(vidOut) 

outFrame_final = im2uint8(outFrame);
                     
writeVideo(vidOut, outFrame_final);

close(vidOut);