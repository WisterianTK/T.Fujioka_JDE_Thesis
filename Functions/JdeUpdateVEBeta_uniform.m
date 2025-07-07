function [new_Beta, Beta_dens] =...
    JdeUpdateVEBeta_uniform(Phi, u_lb,u_ub)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get means for posterior of spatial regulation parameter
%%% INPUT %%%
% Phi: Mixture assignment of neural activation (Categorical dist.)  [H x W x M x K] (K:number of activation classes)
% (i.e., Phi(h,w,m,k) is a probability of pixel (h,w) being in state k for mth condition)
% (i.e., sum(Phi,4) = ones(H,W,M))
% Beta: Spatial regulation parameter in Potts/Ising model       [M x 1]
% u_lb: Lower bound of uniform prior on Beta
% u_ub: Upper bound of uniform prior on Beta

%%% OUTPUT %%%
% new_Beta: Mean of posterior Beta [M x 1] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = size(Phi,3);    % Number of EPs
grid_size = 2000;    % Grid size of density function


new_Beta = zeros(M,1);
Beta_support = linspace(u_lb,u_ub,grid_size);
Beta_dens = zeros(M,grid_size);

% Compute normalization constant of posterior Beta
for m=1:M
    fun = @(beta_m) EvalUnnormalizedDensityBeta_m_uniform(squeeze(Phi(:,:,m,:)), beta_m,u_lb,u_ub);
    % Get unnormalized log density
    log_unnorm_dens = fun(Beta_support);
    % Get log normalization constant
    log_norm_const = logsumexp(log_unnorm_dens);

   % Get log (normalized) density
   log_dens = log_unnorm_dens - log_norm_const;

   Beta_dens(m,:) = exp(log_dens);
end

% % Compute means of posterior Beta
for m=1:M
    new_Beta(m) = sum(Beta_dens(m,:).*Beta_support,"all");
end

end


