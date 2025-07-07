function sigma_h =...
    JdeUpdateVMsigma_h(Sigma_h, mu_h, R, lambda_h, loc_h)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update for posterior parameter on HRF
%%% INPUT %%%
% Sigma_h: Current estimate of covariance   [L_h x L] (L:length of HRF)
% mu_h: Current estimate of mean            [L_h x 1]
% R: Covariance matrix for prior of HRF that controls smoothness of HRF [L x L]
% lambda_vh: exponential distribution prior on v_h
%
%%% OUTPUT %%%
% v_h: Parameter for prior of HRF that controls magnitude of variance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L_h = size(Sigma_h,1);    % Filter length of HRF

% sigma_h = trace((mu_h*mu_h' + Sigma_h)/R)/L_h;  % Without exponential prior
% sigma_h = (-L_h + sqrt(8*lambda_h*trace((mu_h*mu_h' + Sigma_h)/R) + L_h^2))/(4*lambda_h); % With exponential prior
sigma_h = (-L_h + sqrt(8*lambda_h*trace(((mu_h-loc_h)*(mu_h-loc_h)' + Sigma_h)/R) + L_h^2))/(4*lambda_h); % With exponential prior
end