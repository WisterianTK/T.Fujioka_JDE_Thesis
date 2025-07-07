%% JDE on real data

clear 
close all
set(groot,'defaultAxesXGrid','on')
set(groot,'defaultAxesYGrid','on')
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'DefaultLineLineWidth',1)

addpath("Functions")
%
directory_name= "Subjects_for_MATLAB";
subject_name = "CR017";
% subject_name = "CR022";
session_date = "2019-11-13";session_num = "3";
% session_date = "2020-10-07"; session_num = "5";
% session_date = "2019-11-14";session_num = "5";
session_type = "checkerboard";


data_path = fullfile(pwd,directory_name,subject_name,session_date,session_type,session_num,"data.mat");
save_path = fullfile(pwd,"Results_parcel", "without_preprocess","Zero_mean_HRF", "uniform_gamma");

load(data_path)

fs = 1/f_dt;
fc = 0.3;     % cut-off frequency
start_sample = 10;
Y = double(permute(PDI,[2,3,1]));   % Dimension [H x W x N]
Y = Y(:,:,start_sample:end-10); % Cut the last 10 samples
% Y = Y(31:60,51:70,:); % Small patch
% Y = zscore(Y,0,3); % Standardize
% Y = Y./sqrt(var(Y,0,3)); % 04/07
% [b, a] = butter(5,fc/(fs/2),'low');
% Y = reshape(filtfilt(b,a,reshape(Y,[],size(Y,3))')',size(Y,1),size(Y,2),size(Y,3));
% Y = reshape(filloutliers(reshape(Y,[],size(Y,3))',"linear","movmean",200)',125,80,[]);
% Y = Y./sqrt(var(Y,0,3)); % 04/07
% Y = zscore(Y,0,3); % Standardize

stimulus = stim;                    % [N x M], (:,1)=left, (:,2)=center, (:,3)=right
stimulus = stim(start_sample:end-10,:); % Cut the last 10 samples
t_PDI = t_PDI(start_sample:end-10);


% Dimensions
H = size(Y,1);    % Height
W = size(Y,2);    % Width
N = size(Y,3);              % Number of time samples
M = size(stimulus,2);    % Number of EPs
K = 2;                   % Number of activation states
L_h = 35;               % Filter length of HRF  (10.5s)
L_r = 12;               % Filter length of NRF  (4s)

mask_set = {};
mask1 = zeros(H,W);
mask1(21:60,1:25) = 1;
mask2 = zeros(H,W);
mask2(36:60,26:38) = 1;
mask3 = zeros(H,W);
mask3(40:65,42:55) = 1;
mask4 = zeros(H,W);
mask4(30:60,55:78) = 1;

mask_set{1} = mask1;
mask_set{2} = mask2;
mask_set{3} = mask3;
mask_set{4} = mask4;
mask_dims = {};
mask_dims{1} = [40, 25];
mask_dims{2} = [25,13];
mask_dims{3} = [26,14];
mask_dims{4} = [31,24];

onset = (diag(ones(N,1)) - diag(ones(N-1,1),-1))*stimulus;
onset(onset < 0) = 0;
X = zeros(N,L_r+L_h-1,M);
for m=1:M
    X(:,:,m) = toeplitz(onset(:,m),[onset(1,m);zeros(L_r+L_h-2,1)]);
end

%% Set all necessary parameters

fmin = 0.02; % Highest frequency in DCT to be captured as drift
Ldim = ceil(2*N/fs*fmin)+1; % Number of basis
% Ldim = 2*Ldim; % For DST and polynomial

P1 = dctmtx(N);          % DCT
P2 = dstmtx(N);          % DST (not built-in function)
% P1 = P1(1:Ldim/2,:)';    % Take only Ldim/2 basis [N x U/2] (U:number of drift components in total)
P1 = P1(1:Ldim,:)';

P = P1;

% HRF kernel
D = eye(L_h)*-2 + diag(ones(1,L_h-1),1) + diag(ones(1,L_h-1),-1); % Second-order centered Difference matrix

D(1,1) = -50;  % penalize large start and end 
D(end,end) = -50;
R = fs^-4*inv(D'*D); % Covariance of base distribution for HRF
R = R + eye(L_h)*0.001; % For condition number
%% Kernels for NRLs

% SS kernel
lambda_ss = 300; % used
% lambda_ss = 1000;
% lambda_ss = 1.4;
% lambda_ss = 100; %31/05
alpha_ss = 0.84;
kernel = zeros(L_r);
for j=1:L_r
    for k=1:L_r
        kernel(j,k) = lambda_ss*(alpha_ss^(k+j+max(j,k))/2 - alpha_ss^(3*max(j,k))/6);
    end
end

kernel = kernel + eye(L_r)*0.0005*lambda_ss; % For condition number

% % DC kernel
% lambda = 100;
% alpha = 0.8;
% rho = 0.95;
% kernel = zeros(L_r);
% for j=1:L_r
%     for k=1:L_r
%         kernel(j,k) = lambda*(alpha^(k+j))*rho^abs(j-k);
%     end
% end

[u,s,v] = svd(kernel);
loc = 1.4*abs(u(:,1))*0; %ss
% loc = 14*abs(u(:,1)); %ss
% loc = 25*abs(u(:,1)); %dc
loc_r = zeros(L_r,M,K);
loc_r(:,:,1) = repmat(loc,1,M);
loc_r(:,:,2) = zeros(L_r,M);

% Concatenate: gt_kernel [L_r x L_r x M x K]
kernel1 = cat(3,kernel,kernel,kernel);

% For neural state 2 (non-active) under EP1 
kernel_21 = det(kernel)^(1/L_r)*eye(L_r);
% kernel_21 = det(kernel)^(1/L_r)*eye(L_r)*1000;

% For neural state 2 (non-active) under EP2
kernel_22 = det(kernel)^(1/L_r)*eye(L_r);
% kernel_22 = det(kernel)^(1/L_r)*eye(L_r)*1000;

% For neural state 2 (non-active) under EP3
kernel_23 = det(kernel)^(1/L_r)*eye(L_r);
% kernel_23 = det(kernel)^(1/L_r)*eye(L_r)*1000;

kernel2 = cat(3,kernel_21,kernel_22,kernel_23);

Sigma_K = cat(4,kernel1,kernel2);



loc_h = zeros(L_h,1);

%% RUN
seed_range = 1;
num_iter = 100;
for parcel_id = 1:length(mask_set)
    parcel = Y(repmat(mask_set{parcel_id}==1,1,1,N));
    dims = [mask_dims{parcel_id},N];
    parcel = reshape(parcel,dims);
    % parcel = filloutliers(reshape(parcel,[],N)',"linear","movmean",200);
    % parcel = reshape(parcel',dims);

    for j = 1:length(seed_range)
        seed = seed_range(j);
        rng(seed)
    
        % Initialization
        % sigma_b = 1e8*ones(H,W);
        sigma_b = var(parcel,0,3);
        % sigma_h = 0.1;
        sigma_h = 100;%  31/05
        
        Sigma_h = eye(L_h)*0.01;
        % mu_h = sin(pi*(0:(L_h-1))/(L_h-1))';
        % mu_h = 1000*sin(pi*(0:(L_h-1))/(L_h-1))';
        mu_h = gampdf(0:34,3,2)'; mu_h = mu_h*1000/max(mu_h);
        % mu_h = mvnrnd(zeros(L_h,1),sigma_h*R)';
        % mu_h = mu_h/sign(mu_h(abs(mu_h) == max(abs(mu_h))));
        % load(fullfile(pwd,'initializations/muh_init.mat'))
        % mu_h = mu_h/max(mu_h);
        
        % Phi = rand(H,W,M,K);
        Phi = 0.5*ones([size(parcel,[1,2]),M,K]);
        % Phi = Phi./sum(Phi,4);
        % load(fullfile(pwd,"initializations/Phi_init_3.mat"))
        % Phi =  reshape(Phi(repmat(mask_set{parcel_id}==1,1,1,M,K)),dims(1),dims(2),M,K);
        % Phi = Phi(31:60,51:70,:,:);
        Beta = 0.88*ones(M,1);
        
        u_lb = 0.8;
        u_ub = 2;
    
        L_coef = JdeUpdateVML(0*mu_h,0*zeros([size(parcel,[1,2]),L_r*M]),X,parcel,P);
    
        [Sigma_r, mu_r, Sigma_h, mu_h, Phi, Beta, Beta_dens, sigma_b, L_coef, ELBO, sigma_h] = ...
        RunJDE(num_iter, X, parcel, Sigma_h, mu_h, Phi, P, L_coef, Beta, sigma_b, sigma_h, Sigma_K, R, u_lb, u_ub,false,false, 0, loc_r,loc_h);
    
        %%% Save the results from RunJDE at save_path + \kernel_name + 'SNR_' + SNR + '_seed' + seed.mat   %%%
        % Create folder if it doesn't exist
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end
    
        % Construct file name
        filename = sprintf('parcel_%d_seed_%d.mat', parcel_id,seed);
        full_save_path = fullfile(save_path, filename);
    
        % Save relevant variables
        save(full_save_path, ...
            'parcel','X','Sigma_r', 'mu_r', 'Sigma_h', 'mu_h','sigma_h', ...
            'Phi', 'Beta', 'Beta_dens', 'P', ...
            'sigma_b', 'L_coef', 'ELBO','u_lb', 'u_ub', ...
             'seed','Sigma_K','loc_r');
    
    end
end
