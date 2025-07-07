function value = EvalUnnormalizedDensityBeta_m_uniform(Phi_m, beta_m,u_lb,u_ub)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get value of unnormalized log posterior density of Beta evaluated as a
% function of beta_m
%%% INPUT %%%
% Phi_m: Mixture assignment of neural states under mth EP  [H x W x K] (K:number of activation classes)
% (i.e., Phi_m(h,w,k) is a probability of pixel (h,w) being in state k for mth condition)
% (i.e., sum(Phi,3) = ones(H,W))
% beta_m: Spatial regulation parameter in Potts/Ising model for mth EP   
% u_lb: Lower bound of uniform prior on Beta
% u_ub: Upper bound of uniform prior on Beta

%%% OUTPUT %%%
% value: value of unnormalized posterior density of Beta evaluated at beta_m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(Phi_m,1);    % Height
W = size(Phi_m,2);    % Width
K = size(Phi_m,3);  % Number of activation classes

log_value = 0;
for h=1:H
    for w=1:W
        % Find neighbors of current pixel at (h,w)
        neighbor_index = find2Dneighbor(h,w,H,W);

        % Extract Phi values of neighbors (for all states)   [I x K] (I:number of neighbors)
        Phi_m_neighbor = Phi_m(repmat(neighbor_index,1,K) + H*W*(0:K-1));

        % Extract Phi values at (h,w) (for all states)  [K x 1] 
        Phi_m_center = squeeze(Phi_m(h,w,:));
        log_value = log_value + beta_m.*sum(Phi_m_neighbor*Phi_m_center) - log(sum(exp(beta_m'*sum(Phi_m_neighbor,1)),2))';
    end
end

% Add log probability from prior
log_value = log_value - log(u_ub-u_lb) + log(beta_m);

value = log_value;

end


