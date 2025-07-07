function new_sigma_b = ...
    JdeUpdateVMsigma_b(Sigma_h, mu_h, Sigma_r, mu_r, X, Y, P, L_coef, circ_mask)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update for sigma_b
%%% INPUT %%%
% Sigma_h: Current estimate of covariance   [L_h x L] (L_h:length of HRF)
% mu_h: Current estimate of mean HRF            [L_h x 1]
% Sigma_r: Current estimate of set of covariances of NRFs over EPs
%                                           [H x W x (L_r*M) x (L_r*M)] (M:number of EPs, L_r:length of NRF)
% mu_r: Current estimate of set of means of NRFs    [H x W x (L_r*M)]
% X: Design(Toeplitz) matrices based on EPs         [N x (L_h+L_r-1) x M] (N:number of time samples)
% Y: fUS data                                  [H x W x N]
% P: Matrix containing basis vectors for low frequency drift [N x U] (U:number of coefficients)
% L_coef: Set of coefficient vectors for low frequency drift [H x W x U]
% circ_mask: Mask used when generating tensor that contains circshfit of
% covariance matrices.
%
%%% OUTPUT %%%
% new_sigma_b: New estimate of pixel-dependent noise variances [H x W]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(Y,1);  % Height
W = size(Y,2);  % Width
M = size(X,3);  % Number of EPs
N = size(Y,3);    % Number of time samples
L_h = size(mu_h,1); % Filter length of HRF
L_r = size(mu_r,3)/M;    % Filter length of NRF

new_sigma_b = zeros(H*W,1);
Y_vec = reshape(Y,[],N);
L_vec = reshape(L_coef,[], size(L_coef,3));
mu_r_vec = reshape(mu_r, [], size(mu_r,3));
Sigma_r_vec = reshape(Sigma_r,[],size(Sigma_r,3),size(Sigma_r,3));

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

for j=1:H*W
    y_j = Y_vec(j,:)';
    el_j = L_vec(j,:)';
    mu_rj = mu_r_vec(j,:)';
    Sigma_rj = squeeze(Sigma_r_vec(j,:,:));
    new_sigma_b(j) = 1/N*((y_j - 2*exG*mu_rj - P*el_j)'*(y_j - P*el_j)...
        + mu_rj'*exGG*mu_rj + trace(exGG*Sigma_rj));
end
new_sigma_b = reshape(new_sigma_b,H,W);
%% OLD
% circ_mask_reshaped = sparse(boolean(reshape(circ_mask,(L_r+L_h-1)^2,L_h^2)));
% 
% parfor j = 1:H*W
%     [h, w] = ind2sub([H, W], j);
%     RXXR_term1 = 0; % First term in covariance: Expectation of R'*X'*X*R
%     RXXR_term2 = 0; % Second term in covariance
%     sumXexR_j = zeros(N,L_h);   % Sum of X_m*ex[R_mj] for m=1:M
%     Sigma_r_jmm_shift_reshaped = zeros((L_r+L_h-1)^2,L_h^2);
%     Sigma_r_jnm_shift_reshaped = zeros((L_r+L_h-1)^2,L_h^2);
% 
%     for m=1:M
%         % Compute the first term: Expectation of R'*X'*X*R
%         R_mj = squeeze(mu_r(h,w,((m-1)*L_r+1):(m*L_r))); % mean of nrf at jth pixel for mth EP
%         R_mj = toeplitz([R_mj;zeros(L_h-1,1)],[R_mj(1),zeros(1,L_h-1)]); % Toeplitz of size [(L_r+L_h-1) x L_h]
% 
% 
%         XmXm = X(:,:,m)'*X(:,:,m); % Term X(:,:,m)'*X(:,:,m)
%         sumXexR_j = sumXexR_j + X(:,:,m)*R_mj;
% 
%         RXXR_term1_1 = R_mj'*XmXm*R_mj;
% 
%         % Precompute all circularcshifted versions of Sigma_r_jnm
%         % Sigma_r_jmm_shifts = zeros(L_r+L_h-1, L_r+L_h-1, L_h, L_h);
%         Sigma_r_jmm = squeeze(Sigma_r(h,w,((m-1)*L_r+1):(m*L_r),((m-1)*L_r+1):(m*L_r))); % Extract covariance of r_jm
%         % Sigma_r_jmm = zeros(L_r+L_h-1);
%         % Sigma_r_jmm(1:L_r,1:L_r) = Sigma_r(h,w,((m-1)*L_r+1):(m*L_r),((m-1)*L_r+1):(m*L_r)); % Extract covariance of r_jm
% 
%         % for a = 1:L_h
%         %     for b = 1:L_h
%         %         Sigma_r_jmm_shifts(:,:,a,b) = circshift(Sigma_r_jmm, [b-1, a-1])';
%         %     end
%         % end
% 
%         % Matricize: (vectorized circular shifted Sigma_H) x Number of circular shifted versions of Sigma_H  [(L_r+L_h-1)^2 x L_h^2]
%         % Sigma_r_jmm_shift_reshaped = reshape(Sigma_r_jmm_shifts, (L_r+L_h-1)*(L_r+L_h-1), L_h*L_h);  
%         Sigma_r_jmm_shift_reshaped(circ_mask_reshaped) = repmat(reshape(Sigma_r_jmm,[],1),1,L_h^2);
% 
%         % vectorized version of term X(:,:,m)'*X(:,:,n)  [(L_r+L_h-1)^2 x 1]
%         XmXm_vec = reshape(XmXm', [], 1);  
% 
%         % Compute trace(  X(:,:,m)'*X(:,:,n)*circshift(Sigma_r_jnm, b-1,a-1)) for all b,a=1:L_h 
%         RXXR_term1_2 = reshape(XmXm_vec' * Sigma_r_jmm_shift_reshaped, L_h, L_h);
% 
%         % RXXR_term1_2 = zeros(L_h);
%         % for a=1:L_h
%         %     for b=1:L_h
%         %         % RXXR_term1_2(a,b) = trace(XmXm*circshift(Sigma_r_jm,[b-1,a-1]));
%         %         RXXR_term1_2(a,b) = sum(sum(XmXm.*(circshift(Sigma_r_jmm,[b-1,a-1])'))); % Note: trace(A*B) = sum(sum(A.*B'))
%         %     end
%         % end
% 
%         RXXR_term1 = RXXR_term1 + RXXR_term1_1 + RXXR_term1_2;
% 
%         % Compute the second term
%         for n=1:(m-1)
%             R_nj = squeeze(mu_r(h,w,((n-1)*L_r+1):(n*L_r))); % mean of nrf at jth pixel for nth EP
%             R_nj = toeplitz([R_nj;zeros(L_h-1,1)],[R_nj(1),zeros(1,L_h-1)]); % Toeplitz of size [(L_r+L_h-1) x L_h]
% 
%             XmXn = X(:,:,m)'*X(:,:,n); % Term X(:,:,m)'*X(:,:,n)
% 
%             RXX_R_term2_1 = R_mj'*XmXn*R_nj;
% 
%             % Precompute all circularcshifted versions of Sigma_r_jnm
%             Sigma_r_jnm = squeeze(Sigma_r(h,w,((n-1)*L_r+1):(n*L_r),((m-1)*L_r+1):(m*L_r))); % Extract covariance of R_jn and R_jm
%             % Sigma_r_jnm_shifts = zeros(L_r+L_h-1, L_r+L_h-1, L_h, L_h);           
%             % Sigma_r_jnm = zeros(L_r+L_h-1);
%             % Sigma_r_jnm(1:L_r,1:L_r) = Sigma_r(h,w,((n-1)*L_r+1):(n*L_r),((m-1)*L_r+1):(m*L_r)); % Extract covariance of R_jn and R_jm
%             % for a = 1:L_h
%             %     for b = 1:L_h
%             %         Sigma_r_jnm_shifts(:,:,a,b) = circshift(Sigma_r_jnm, [b-1, a-1])';
%             %     end
%             % end
% 
%             % Matricize: (vectorized circular shifted Sigma_H) x Number of circular shifted versions of Sigma_H  [(L_r+L_h-1)^2 x L_h^2]
%             % Sigma_r_jnm_shift_reshaped = reshape(Sigma_r_jnm_shifts, (L_r+L_h-1)*(L_r+L_h-1), L_h*L_h);  
%             Sigma_r_jnm_shift_reshaped(circ_mask_reshaped) = repmat(reshape(Sigma_r_jnm,[],1),1,L_h^2);
% 
%             % vectorized version of term X(:,:,m)'*X(:,:,n)  [(L_r+L_h-1)^2 x 1]
%             XmXn_vec = reshape(XmXn', [], 1);  
% 
%             % Compute trace(  X(:,:,m)'*X(:,:,n)*circshift(Sigma_r_jnm, b-1,a-1)) for all b,a=1:L_h 
%             RXXR_term2_2 = reshape(XmXn_vec' * Sigma_r_jnm_shift_reshaped, L_h, L_h);
% 
%             % RXXR_term2_2 = zeros(L_h);
%             % for a=1:L_h
%             %     for b=1:L_h
%             %         % term2_2(a,b) = trace(XmXn*circshift(Sigma_r_jnm,[b-1,a-1]));
%             %         RXXR_term2_2(a,b) = sum(sum(XmXn.*(circshift(Sigma_r_jnm,[b-1,a-1])'))); % Note: trace(A*B) = sum(sum(A.*B'))
%             %     end
%             % end
% 
%             RXXR_term2 = RXXR_term2 + RXX_R_term2_1 + RXXR_term2_2;
%         end
%     end
%     RXXR_term = RXXR_term1 + (RXXR_term2 + RXXR_term2');
% 
%     y_j = squeeze(Y(h,w,:));
%     l_j = squeeze(L_coef(h,w,:));
%     term1 = (y_j - P*l_j)'*(y_j - P*l_j);
%     term2 = -2*(sumXexR_j*mu_h)'*(y_j - P*l_j) ;
%     term3 = mu_h'*RXXR_term*mu_h + trace(RXXR_term*Sigma_h);
%     new_sigma_b(j) = (term1+term2+term3)/N;
% end

end


