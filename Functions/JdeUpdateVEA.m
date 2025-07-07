function [new_Sigma_r, new_mu_r] =...
    JdeUpdateVEA(Sigma_h, mu_h, X, Y, P, L_coef, sigma_b, Sigma_K, Phi, loc_r)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update for posterior parameters on NRFs
%%% INPUT %%%
% Sigma_h: Current estimate of covariance   [L_h x L_h] (L_h:length of HRF)
% mu_h: Current estimate of mean            [L_h x 1]
% X: Design(Toeplitz) matrices based on EPs [N x (L_h+L_r-1) x M] (N:number of time samples, L_r:length of NRF, M:number of EPs)
% Y: fUS data                              [H x W x N]
% P: Matrix containing basis vectors for low frequency drift [N x U] (U:number of coefficients)
% L_coef: Set of coefficient vectors for low frequency drift [H x W x U]
% sigma_b: Pixel-dependent noise variance   [H x W]
% Sigma_K: Covariance matrices for each activation class and EP [L_r x L_r x M x K]
% Phi: Mixture assignment of neural activation (Categorical dist.)  [H x W x M x K] (K:number of activation classes)
% (i.e., Phi(h,w,m,k) is a probability of pixel (h,w) being in state k for mth condition)
% (i.e., sum(Phi,4) = ones(H,W,M))
% loc_r: prior mean of NRF [L_r x M x K]

%%% OUTPUT %%%
% new_Sigma_r: New estimate of covariances of NRFs over EPs [H x W x (L_r*M) x (L_r*M)] 
% new_mu_r: New estimate of mean of NRFs         [H x W x (L_r*M)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(Y,1);    % Height
W = size(Y,2);    % Width
M = size(X,3);    % Number of EPs
N = size(Y,3);    % Number of time samples
K = size(Phi,4);  % Number of activation classes
L_r = size(Sigma_K,1);  % Filter length of NRF
L_h = size(mu_h,1);     % Filter length of HRF

new_Sigma_r = zeros(H,W,L_r*M,L_r*M);   % Initialize new estimate for Sigma_a
new_Sigma_r = reshape(new_Sigma_r,H*W,L_r*M,L_r*M);
new_mu_r = zeros(H,W,L_r*M);        % Initialize new estimate for mu_a
new_mu_r = reshape(new_mu_r,H*W,L_r*M);

Y_vec = reshape(Y,[],N);
L_vec = reshape(L_coef,[], size(L_coef,3));
Phi_vec = reshape(Phi,[],M,K);
loc_r_vec = reshape(loc_r,[],1);

% Precompute E[G]
exG = zeros(N, M*L_r);
H_toep = toeplitz([mu_h; zeros(L_r-1,1)], [mu_h(1);zeros(L_r-1,1)]);
for m=1:M
    exG(:,(m-1)*L_r+1:m*L_r) = X(:,:,m)*H_toep;
end

% Precompute all circularcshifted versions of Simga_H
Sigma_H = zeros(L_r+L_h-1);
Sigma_H(1:L_h,1:L_h) = Sigma_h; % Used in trace term
Sigma_H_shifts = zeros(L_r+L_h-1, L_r+L_h-1, L_r, L_r);
for a = 1:L_r
    for b = 1:L_r
        Sigma_H_shifts(:,:,a,b) = circshift(Sigma_H, [b-1, a-1])'; 
    end
end

% Matricize: (vectorized circular shifted Sigma_H) x Number of circular shifted versions of Sigma_H  [(L_r+L_h-1)^2 x L_r^2]
Sigma_H_shift_reshaped = reshape(Sigma_H_shifts, (L_r+L_h-1)*(L_r+L_h-1), L_r*L_r);  

% Precompute E[G'G]      [(L_r*M) x (L_r*M)]
exGG = zeros(L_r*M);
for m = 1:M
    for n = 1:M
        % trace_term = zeros(L_r);
        XmXn = X(:,:,m)'*X(:,:,n); 
        % vectorized version of term X(:,:,m)'*X(:,:,n)  [(L_r+L_h-1)^2 x 1]
        XmXn_vec = reshape(XmXn, [], 1);  

        % Compute trace(  X(:,:,m)'*X(:,:,n)*circshift(Sigma_H,b-1,a-1))
        trace_term = reshape(Sigma_H_shift_reshaped' * XmXn_vec, L_r, L_r);
        
        exGG(L_r*(m-1)+1:L_r*m,L_r*(n-1)+1:L_r*n) = H_toep'*XmXn*H_toep + trace_term;
    end
end


blk_inv_Sigma_K = zeros(L_r*M,L_r*M,K); % Each slice contains block diagonal matrix
for k=1:K
    for m=1:M
        blk_inv_Sigma_K(L_r*(m-1)+1:L_r*m,L_r*(m-1)+1:L_r*m,k) = inv(squeeze(Sigma_K(:,:,m,k)));
    end
end

parfor j = 1:H*W
    y_j = squeeze(Y_vec(j,:))';
    el_j = squeeze(L_vec(j,:))';
    weighted_blk_inv_Sigma_K = zeros(L_r*M,L_r*M,K); % Each diagonal block is weighted by corresponding Phi
    for k=1:K
        for m=1:M
            Phi_jmk = Phi_vec(j,m,k);
            weighted_blk_inv_Sigma_K(L_r*(m-1)+1:L_r*m,L_r*(m-1)+1:L_r*m,k) = Phi_jmk*blk_inv_Sigma_K(L_r*(m-1)+1:L_r*m,L_r*(m-1)+1:L_r*m,k);
        end
    end

    % Compute Sigma_r for r_j 
    % new_Sigma_r(h,w,:,:) = inv(exGGammaG + sum(weighted_blk_inv_Sigma_K,3));
    new_Sigma_r(j,:,:) = inv(1/sigma_b(j)*exGG + sum(weighted_blk_inv_Sigma_K,3));

    % Update mu_r for r_j
    % new_mu_r(h,w,:) = squeeze(new_Sigma_r(h,w,:,:))*(exG*Gamma*(y_j - P*l_j)+reshape(weighted_blk_inv_Sigma_K,L_r*M,[])*loc_r_vec);
    new_mu_r(j,:) = squeeze(new_Sigma_r(j,:,:))*(1/sigma_b(j)*exG'*(y_j - P*el_j)+reshape(weighted_blk_inv_Sigma_K,L_r*M,[])*loc_r_vec);
end
new_Sigma_r = reshape(new_Sigma_r,H,W,L_r*M,L_r*M);
new_mu_r = reshape(new_mu_r,H,W,L_r*M);

end
