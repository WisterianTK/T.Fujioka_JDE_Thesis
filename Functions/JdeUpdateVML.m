function new_L_coef = ...
    JdeUpdateVML(mu_h, mu_r, X, Y, P)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update for hyperparameters for low frequency drift components
%%% INPUT %%%
% mu_h: Current estimate of mean HRF                [L_h x 1]  (L_h:length of HRF)
% mu_r: Current estimate of means of NRFs     [H x W x (L_r*M)] (M:number of EPs, L_r:length of NRF)
% X: Design(Toeplitz) matrices based on EPs [N x (L_h+L_r-1) x M]  (N:number of time samples)
% Y: fUS data                              [H x W x N]
% P: Matrix containing basis vectors for low frequency drift [N x U] (U:number of coefficients)
%
%%% OUTPUT %%%
% new_L_coef: New estimate of set of coefficient vectors for low frequency drift [H x W x U]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(Y,1);  % Height
W = size(Y,2);  % Width
U = size(P,2);  % Number of coefficients for low frequency drift
M = size(X,3);  % Number of EPs
N = size(Y,3);  % Number of time samples
L_h = size(mu_h,1); % Filter length of HRF
L_r = size(mu_r,3)/M;    % Filter length of NRF


new_L_coef = zeros(H*W,U);

Y_vec = reshape(Y,[],N);
mu_r_vec = reshape(mu_r, [], size(mu_r,3));

pseudo_inv_P = pinv(P); % Precompute

% Precompute E[G]
exG = zeros(N, M*L_r);
H_toep = toeplitz([mu_h; zeros(L_r-1,1)], [mu_h(1);zeros(L_r-1,1)]);
for m=1:M
    exG(:,(m-1)*L_r+1:m*L_r) = X(:,:,m)*H_toep;
end

for j=1:H*W
    y_j = Y_vec(j,:)';
    mu_rj = mu_r_vec(j,:)';
    new_L_coef(j,:) = pseudo_inv_P*(y_j - exG*mu_rj);
end
new_L_coef = reshape(new_L_coef,H,W,U);

%% OLD
% for h=1:H
%     for w=1:W
%         sumXexR_j = zeros(N,L_h);   % Sum of X_m*ex[R_mj] for m=1:M
%         for m=1:M
%             R_mj = squeeze(mu_r(h,w,((m-1)*L_r+1):(m*L_r))); % mean of nrf at jth pixel for mth EP
%             R_mj = toeplitz([R_mj;zeros(L_h-1,1)],[R_mj(1),zeros(1,L_h-1)]); % Toeplitz of size [(L_r+L_h-1) x L_h]
%             sumXexR_j = sumXexR_j + X(:,:,m)*R_mj;
%         end
%         % Closed-form solution of alternating least square for l_j
%         % new_L_coef(h,w,:) = (P'*P)\P'*(squeeze(Y(h,w,:)) - sumXexR_j*mu_h);
%         new_L_coef(h,w,:) = pseudo_inv_P*(squeeze(Y(h,w,:)) - sumXexR_j*mu_h);
%     end
% end

end

