function [gt_Y, Y, X, gt_NRF, gt_L_coef] = ...
    JdeGenSynthfUS(NAS_pat, N, P, fs, gt_sigma_b,...
                      gt_hrf, gt_nrf, NstimBlock)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update for variance within HRF-group
%%% INPUT %%%
% NAS_pat: Neural activation state pattern [H x W x M] (M:number of EPs)
% N: Number of time samples
% Ldim: Number of DCT basis 
% fs: Sampling frequency
% gt_sigma: Groundtruth of noise variance
% gt_hrf: Ground truth HRF                  [L_h x 1] (L_h:length of HRF)
% gt_nrf:  Ground truth NRFs for each EP    [L_r x M] (L_r:length of NRF)
% NstimBlock: Number of block stimuli per condition
%
%%% OUTPUT %%%
% Y: Synthetic fUS data                             [H x W x N] (N:number of time samples)
% X: Design(Toeplitz) matrices based on EPs         [N x (L_h + L_r -1) x M] 
% gt_Nrf: Groundtruth neural response levels        [H x W x L_r x M]
% gt_L_coef: Set of coefficient vectors for low frequency drift (groundtruth) [H x W x U] (U:number of basis components)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(NAS_pat,1);    % Height
W = size(NAS_pat,2);    % Width
M = size(NAS_pat,3);    % Number of EPs
L_h = size(gt_hrf,1);   % Length of HRF
L_r = size(gt_nrf,1);   % Length of NRF

%% Synthetic NRF
gt_NRF = zeros(H,W,L_r,M);
for m=1:M
    % Copy NRF for activated region
    gt_NRF(:,:,:,m) = reshape(gt_nrf(:,m),1,1,L_r).*(NAS_pat(:,:,m) == 1);

    % Add noise
    gt_NRF(:,:,:,m) = gt_NRF(:,:,:,m) + 0.015*randn(H,W,L_r);

    % % Add noise
    % gt_NRF(:,:,:,m) = gt_NRF(:,:,:,m) + 0.02*randn(H,W,L_r).*(NAS_pat(:,:,m) == 1);

    % Add noise
    % gt_NRF(:,:,:,m) = gt_NRF(:,:,:,m) + 0.03*randn(H,W,L_r);
end

%% Synthetic experimental paradigms
% Generate random ON durations
% min_on = round(fs); % Minimum ON duration 1s
min_on = 1; % Minimum ON duration is one sample
% max_on = round(3*fs); % Maximum ON duration 3s
% max_on = 2; % Maximum ON duration two samples
max_on = 1; % Maximum ON duration one sample
on_durations = randi([min_on max_on], M, NstimBlock);

% Generate random OFF durations between stimuli
min_off = round(1*fs); % 05/03 specificity of activation
max_off = round(4*fs); % 05/03
% min_off = round(8*fs*2*2); % 05/03 specificity of activation
% max_off = round(10*fs*2*2); % 05/03
% min_off = round(8*fs); % 05/03 specificity of activation
% max_off = round(10*fs); % 05/03

off_durations = randi([min_off max_off], M, NstimBlock-1);

total_length = max(sum(on_durations,2)) + max(sum(off_durations,2));

% Construct stimulus sequence
EP = zeros(M, total_length);

for m = 1:M
    position = 1 + (m-1)*max_on; % Start position (avoid overlapping EP)

    for i = 1:NstimBlock
        % Set ON period
        EP(m, position:position + on_durations(m,i) - 1) = 1;
        position = position + on_durations(m,i);
        
        % Set OFF period (if not the last one)
        if i < NstimBlock
            position = position + off_durations(m,i);
        end
    end
end
total_length = size(EP,2);

% Fit to the length we want
if total_length > N
    EP = EP(:,1:N);
elseif N > total_length
    EP = cat(2,EP, zeros(M,N-total_length));
end

% % I noticed that EPs often coincide in this generation method: circshift
% EP(2,:) = circshift(EP(2,:),50);

%% Synthetic Design matrix [N x (L_h+L_r-1) x M]
X = zeros(N,L_h+L_r-1,M);
for m=1:M
    X(:,:,m) = toeplitz(EP(m,:), [EP(m,1), zeros(1,L_h + L_r - 2)]);
end

%% Synthetic low frequency drift (Ciuciu 2003: unspervised robust nonpara..)
% i.e., Breathing and cardiac pulses are aliased since the sampling
% frequency of the data is below Nyquist's bound.
% Formula for number of DCT basis: Ldim = ceil(2*Ns*fmin) + 1
% where fmin: the lowest frequency component attributable to drift.
% DCT-II is used. (orthogonal)
% P1 = dctmtx(N);          % DCT
% P2 = dstmtx(N);          % DST (not built-in function)
% P1 = P1(1:Ldim/2,:)';    % Take only Ldim/2 basis [N x U/2] (U:number of drift components in total)
% P2 = P2(2:Ldim/2,:)';    % Take only Ldim/2 - 1 basis [N x (U/2-1)] (-1 to exculede DC term)
% P3 = sqrt(2/N)*linspace(0,1,N)';    % First-order polynomial
% P = [P1,P2,P3];

Ldim = size(P,2);
% Random generation of coefficients for orthonormal basis (factor of 50 and second term to mimic fUS data)
gt_L_coef = 5*rand([H,W,Ldim]) + 5*cat(3,ones(H,W), zeros(H,W,Ldim-1));
% gt_L_coef = 5000*rand([H,W,Ldim]) + 500000*cat(3,ones(H,W), zeros(H,W,Ldim-1)); % Realistic

%% Measurement noise (From 2008 Ciuciu) (The same rho and sigma for all pixels)
noise = sqrt(gt_sigma_b).*randn(H,W,N);

%% Synthetic fUS signals [H x W x N]
gt_Y = zeros(H,W,N);
Y = zeros(H,W,N);
H_toeplitz = toeplitz([gt_hrf;zeros(L_r-1,1)],[gt_hrf(1),zeros(1,L_r-1)]); % [(L_h+L_r-1) x L_r]
for h=1:H
    for w=1:W
        for m=1:M
            % Convolution for each EP
            Y(h,w,:) = squeeze(Y(h,w,:)) + X(:,:,m)*H_toeplitz*squeeze(gt_NRF(h,w,:,m));
            gt_Y(h,w,:) = Y(h,w,:);
        end
        Y(h,w,:) = squeeze(Y(h,w,:)) + P*squeeze(gt_L_coef(h,w,:)) + squeeze(noise(h,w,:));
    end
end

end
