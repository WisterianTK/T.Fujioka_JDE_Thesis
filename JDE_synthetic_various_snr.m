% Run JDE for variaous SNRs
clear 
close all

% Directory to save data
save_path = fullfile(pwd,'\synthetic_various_snr',"uniform_sine");

% Add path to function directory
addpath(fullfile(pwd,"Functions"))

% Load ground truth for dimention setting
data_path = fullfile(pwd,"Ground_truth_sets");
load(fullfile(data_path,"Ground_truth_for_various_SNR_1.mat"))

H = size(NAS_pat,1);    % Height
W = size(NAS_pat,2);    % Width
N = 240;                % Number of time samples
M = size(NAS_pat,3);    % Number of EPs
K = 2;                  % Number of activation states

L_h = size(gt_hrf,1);               % Filter length of HRF (should be shorter than stimulus interval)
L_r = size(gt_nrf,1);               % Filter length of NRF 

fs = 4;            % Sampling frequency
NstimBlock = 25;

fmin = 1/40; % Highest frequency in DCT to be captured as drift
Ldim = ceil(2*N/fs*fmin)+1; % Number of basis

P1 = dctmtx(N);          % DCT
P1 = P1(1:Ldim,:)';    % Take only Ldim/2 basis [N x U] (U:number of drift components in total)
P = P1;

% Kernel_set_name = {"ss","dc","tc"};
Kernel_set_name = {"ss"};

%% Set range of seeds

% seed_range = (1:10)';  % Note this takes a while
seed_range = 1;
SNR_range = (20:-5:-5)';

%% Smooth kernel
D = eye(L_h)*-2 + diag(ones(1,L_h-1),1) + diag(ones(1,L_h-1),-1); % Second-order centered Difference matrix

D(1,1) = -50; D(end,end) = -50; % penalize large start and end 
R = fs^-4*inv(D'*D); % Covariance of base distribution for HRF

%% Kernels for NRLs

% SS kernel
lambda_ss = 1;
alpha_ss = 0.84;
kernel_ss = zeros(L_r);
for j=1:L_r
    for k=1:L_r
        kernel_ss(j,k) = lambda_ss*(alpha_ss^(k+j+max(j,k))/2 - alpha_ss^(3*max(j,k))/6);
    end
end

kernel_ss = kernel_ss+ eye(L_r)*0.0005*lambda_ss;

% Concatenate [L_r x L_r x M x K]
kernel1 = cat(3,kernel_ss,kernel_ss);

% For neural state 2 (non-active) under EP1 
kernel_21 = det(kernel_ss)^(1/L_r)*eye(L_r);

% For neural state 2 (non-active) under EP2
kernel_22 = det(kernel_ss)^(1/L_r)*eye(L_r);

kernel2 = cat(3,kernel_21,kernel_22);

Sigma_K = cat(4,kernel1,kernel2);

Kernel_set = {Sigma_K};
location_set = {zeros(L_r,M,K)};
%% 
num_iter = 100;
for data_set = 1:3
    load(fullfile(data_path,sprintf("Ground_truth_for_various_SNR_%d.mat",data_set)))

    for kernel_id = 1:1
        kernel_name = Kernel_set_name{kernel_id};
        Sigma_K = Kernel_set{kernel_id};
        loc_r = location_set{kernel_id};
    
        for i = 1:length(SNR_range)
            gt_sigma_b = sigma_range(i);
            SNR = SNR_range(i);
        
            for j = 1:length(seed_range)
                seed = seed_range(j);
                rng(seed)
                Y = gt_Y + sqrt(gt_sigma_b)*randn(size(gt_Y));
        
                % Initialization
                sigma_b = 2*gt_sigma_b*ones(H,W); % Twice of ground Truth
                sigma_h = 1;                
                Sigma_h = eye(L_h);
                mu_h = sin(pi*1/(L_h-1)*(0:(L_h-1)))';
                loc_h = mu_h*0;
                Phi = 0.5*ones(H,W,M,K);
                Beta = 0.88*ones(M,1);
                
                u_lb = 0.6;
                u_ub = 2;
    
                L_coef = zeros(H,W,Ldim);
    
                [Sigma_r, mu_r, Sigma_h, mu_h, Phi, Beta, Beta_dens, sigma_b, L_coef, ELBO, sigma_h] = ...
                RunJDE(num_iter, X, Y, Sigma_h, mu_h, Phi, P, L_coef, Beta, sigma_b, sigma_h, Sigma_K, R, u_lb, u_ub,false,false, 0,loc_r,loc_h);
    
                %%% Save the results from RunJDE at save_path + 'SNR_' + SNR + 'dB_seed' + seed + .mat   %%%
                % Create folder if it doesn't exist
                output_folder = fullfile(save_path, sprintf("set_%d",data_set));
                if ~exist(output_folder, 'dir')
                    mkdir(output_folder);
                end
    
                % Construct file name
                filename = sprintf('SNR_%ddB_seed_%d.mat', SNR, seed);
                full_save_path = fullfile(output_folder, filename);
    
                % Save relevant variables
                save(full_save_path, ...
                    'gt_Y','gt_hrf','gt_nrf','gt_nrfs','Y','X','Sigma_r', 'mu_r', 'Sigma_h', 'mu_h', ...
                    'Phi', 'Beta', 'Beta_dens', 'P', ...
                    'sigma_b', 'sigma_h', 'L_coef', 'ELBO','u_lb', 'u_ub', ...
                    'SNR', 'seed', 'kernel_name','Sigma_K','loc_r','loc_h');
    
            end
        end
    end
end





        






