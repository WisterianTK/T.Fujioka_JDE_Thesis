function value = EvalNumeratorBeta_m(beta_m, Phi)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function that evaluates the numerator of posterior of beta_m
%%% INPUT %%%
% beta_m: Spatial regulartion constant for Ising/Potts model
% Phi   : Posterior soft assignment [H x W x K]
%
%%% OUTPUT %%%
% value: Value of the numerator of posterior evaluated at beta_m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(Phi,1);    % Height
W = size(Phi,2);    % Width
K = size(Phi,3);    % Number of states (classes)

value = 0;
for h = 1:H
    for w = 1:W
        neighbor_index = find2Dneighbor(h,w,H,W);

        % Extract Phi values of neighbors (for all EPs)   [I x K] (I:number of neighbors)
        Phi_neighbor = Phi(repmat(neighbor_index,1,K) + H*W*(0:K-1));

        % Extract Phi values of (h,w) pixels [K x 1]
        Phi_center = squeeze(Phi(h,w,:));

        Phi_prodsum =  sum(Phi_neighbor,1)*Phi_center;

        value = value + (beta_m*Phi_prodsum) - log((sum(exp(beta_m*sum(Phi_neighbor,1)),"all")));
    end
end
% value = exp(value);
end



        


