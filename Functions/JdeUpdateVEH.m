function [new_Sigma_h, new_mu_h] =...
    JdeUpdateVEH(Sigma_r, mu_r, X, sigma_h, R, Y, P, L_coef, sigma_b, circ_mask, iter, loc_h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update for posterior parameters on HRF
%%% INPUT %%%
% Sigma_r: Current estimate of set of covariances of NRFs over EPs
%                                           [H x W x (L_r*M) x (L_r*M)] (M:number of EPs, L_r:length of NRF)
% mu_r: Current estimate of set of means of NRFs    [H x W x (L_r*M)]
% X: Design(Toeplitz) matrices based on EPs         [N x (L_h+L_r-1) x M] (N:number of time samples, L_h:length of HRF)
% sigma_h: Scaling parameters for prior on HRF
% R: Covariance matrix for prior of HRF that controls smoothness of HRF [L x L]
% Y: fUS data                              [H x W x N]
% P: Matrix containing basis vectors for low frequency drift [N x U] (U:number of coefficients)
% L_coef: Set of coefficient vectors for low frequency drift [H x W x U]
% sigma_b: Pixel-dependent noise variances [H x W]
% circ_mask: Mask used when generating tensor that contains circshfit of
% covariance matrices.
%
%%% OUTPUT %%%
% new_Sigma_h: New estimate of covariance of HRF [L x L]
% new_mu_h: New estimate of mean of HRF         [L x 1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(Y,1);    % Height
W = size(Y,2);    % Width
M = size(X,3);    % Number of EPs
L_r = size(mu_r,3)/M;    % Filter length of NRF
L_h = size(X,2) - L_r + 1; % Filter length of HRF


new_mu_h = zeros(L_h,1); % Preallocate new estimate for mu_h
term1 = zeros(L_h); % First term in covariance: Expectation of R'*X'*Gamma*X*R
term2 = zeros(L_h); % Second term in covariance

circ_mask_reshaped = sparse(boolean(reshape(circ_mask,(L_r+L_h-1)^2,L_h^2)));
% circ_mask_reshaped = (boolean(reshape(circ_mask,(L_r+L_h-1)^2,L_h^2)));

% Update Sigma_h and mu_h
parfor j = 1:H*W
    Sigma_r_jmm_shift_reshaped = zeros((L_r+L_h-1)^2,L_h^2);
    Sigma_r_jnm_shift_reshaped = zeros((L_r+L_h-1)^2,L_h^2);
    [h, w] = ind2sub([H, W], j);
    Gamma = 1/sigma_b(j); % Precision for noise 
    y_j = squeeze(Y(h,w,:));
    l_j = squeeze(L_coef(h,w,:));
    for m = 1:M
        % Compute the first term: Expectation of R'*X'*Gamma*X*R
        R_mj = squeeze(mu_r(h,w,((m-1)*L_r+1):(m*L_r))); % mean of nrf at jth pixel for mth EP
        R_mj = toeplitz([R_mj;zeros(L_h-1,1)],[R_mj(1),zeros(1,L_h-1)]); % Toeplitz of size [(L_r+L_h-1) x L_h]
        
        XmGXm = X(:,:,m)'*Gamma*X(:,:,m); % Term X(:,:,m)'*Gamma*X(:,:,m)

        term1_1 = R_mj'*XmGXm*R_mj;

        % Precompute all circularcshifted versions of Sigma_r_jnm
        % Sigma_r_jmm_shifts = zeros(L_r+L_h-1, L_r+L_h-1, L_h, L_h);
        % Sigma_r_jmm_shift_reshaped = zeros((L_r+L_h-1)^2,L_h^2);
        Sigma_r_jmm = squeeze(Sigma_r(h,w,((m-1)*L_r+1):(m*L_r),((m-1)*L_r+1):(m*L_r))); % Extract covariance of r_jm
        % Sigma_r_jmm = zeros(L_r+L_h-1);
        % Sigma_r_jmm(1:L_r,1:L_r) = Sigma_r(h,w,((m-1)*L_r+1):(m*L_r),((m-1)*L_r+1):(m*L_r)); % Extract covariance of r_jm
        % for a = 1:L_h
        %     for b = 1:L_h
        %         Sigma_r_jmm_shifts(:,:,a,b) = circshift(Sigma_r_jmm, [b-1, a-1])';
        %     end
        % end

        % Sigma_r_jmm_shifts(boolean(circ_mask)) = repmat(Sigma_r_jmm,1,L_h,L_h);

        % Matricize: (vectorized circular shifted Sigma_H) x Number of circular shifted versions of Sigma_H  [(L_r+L_h-1)^2 x L_h^2]
        % Sigma_r_jmm_shift_reshaped = reshape(Sigma_r_jmm_shifts, (L_r+L_h-1)*(L_r+L_h-1), L_h*L_h);  
        Sigma_r_jmm_shift_reshaped(circ_mask_reshaped) = repmat(reshape(Sigma_r_jmm,[],1),1,L_h^2);

        % vectorized version of term X(:,:,m)'*Gamma*X(:,:,n) with transpose [(L_r+L_h-1)^2 x 1]
        XmGXm_vec = reshape(XmGXm', [], 1);  

        % Compute trace(  X(:,:,m)'*Gamma*X(:,:,n)*circshift(Sigma_r_jnm, b-1,a-1)) for all b,a=1:L_h 
        term1_2 = reshape(XmGXm_vec'*Sigma_r_jmm_shift_reshaped, L_h, L_h);


        % term1_2 = zeros(L_h);
        % for a=1:L_h
        %     for b=1:L_h
        %         % term1_2(a,b) = trace(XmGXm*circshift(Sigma_r_jm,[b-1,a-1]));
        %         term1_2(a,b) = sum(sum(XmGXm.*(circshift(Sigma_r_jmm,[b-1,a-1])'))); % Note: trace(A*B) = sum(sum(A.*B'))
        %     end
        % end

        term1 = term1 + term1_1 + term1_2;

        % Compute the second term
        for n=1:(m-1)
            R_nj = squeeze(mu_r(h,w,((n-1)*L_r+1):(n*L_r))); % mean of nrf at jth pixel for nth EP
            R_nj = toeplitz([R_nj;zeros(L_h-1,1)],[R_nj(1),zeros(1,L_h-1)]); % Toeplitz of size [(L_r+L_h-1) x L_h]
            XmGXn = X(:,:,m)'*Gamma*X(:,:,n); % Term X(:,:,m)'*Gamma*X(:,:,n)

            term2_1 = R_mj'*XmGXn*R_nj;
            
            % Precompute all circularcshifted versions of Sigma_r_jnm
            % Sigma_r_jnm_shifts = zeros(L_r+L_h-1, L_r+L_h-1, L_h, L_h);
            % Sigma_r_jnm_shift_reshaped = zeros((L_r+L_h-1)^2,L_h^2);
            Sigma_r_jnm = squeeze(Sigma_r(h,w,((n-1)*L_r+1):(n*L_r),((m-1)*L_r+1):(m*L_r))); % Extract covariance of R_jn and R_jm
            % Sigma_r_jnm = zeros(L_r+L_h-1);
            % Sigma_r_jnm(1:L_r,1:L_r) = Sigma_r(h,w,((n-1)*L_r+1):(n*L_r),((m-1)*L_r+1):(m*L_r)); % Extract covariance of R_jn and R_jm
            % for a = 1:L_h
            %     for b = 1:L_h
            %         Sigma_r_jnm_shifts(:,:,a,b) = circshift(Sigma_r_jnm, [b-1, a-1])';
            %     end
            % end
            % Matricize: (vectorized circular shifted Sigma_H) x Number of circular shifted versions of Sigma_H  [(L_r+L_h-1)^2 x L_h^2]
            % Sigma_r_jnm_shift_reshaped = reshape(Sigma_r_jnm_shifts, (L_r+L_h-1)*(L_r+L_h-1), L_h*L_h);  
            Sigma_r_jnm_shift_reshaped(circ_mask_reshaped) = repmat(reshape(Sigma_r_jnm,[],1),1,L_h^2);

            % vectorized version of term X(:,:,m)'*Gamma*X(:,:,n) with transpose  [(L_r+L_h-1)^2 x 1]
            XmGXn_vec = reshape(XmGXn', [], 1);  

            % Compute trace(  X(:,:,m)'*Gamma*X(:,:,n)*circshift(Sigma_r_jnm, b-1,a-1)) for all b,a=1:L_h 
            term2_2 = reshape(XmGXn_vec'*Sigma_r_jnm_shift_reshaped, L_h, L_h);

            % term2_2 = zeros(L_h);
            % for a=1:L_h
            %     for b=1:L_h
            %         % term2_2(a,b) = trace(XmGXn*circshift(Sigma_r_jnm,[b-1,a-1]));
            %         term2_2(a,b) = sum(sum(XmGXn.*(circshift(Sigma_r_jnm,[b-1,a-1])'))); % Note: trace(A*B) = sum(sum(A.*B'))
            %     end
            % end

            term2 = term2 + term2_1 + term2_2;
        end
        % Compute mu_h part
        new_mu_h = new_mu_h + R_mj'*X(:,:,m)'*Gamma*(y_j-P*l_j);
    end
end

% Account for transposed term
term2 = term2 + term2';

new_Sigma_h = inv(inv(R)/sigma_h + term1 + term2);
new_mu_h = new_Sigma_h*(new_mu_h + 1/sigma_h*(R\loc_h));

%% Normalization
if iter > 0
    % Normalize HRF
    [norm_const, peak_id]  = max(abs(new_mu_h));    % Extract maximum magnitude for each group
    if norm_const > 0.1
        % Extract sign of peak values
        sign_norm = sign(new_mu_h(peak_id));
    
        % Normalize new_mu_h
        new_mu_h = sign_norm.*new_mu_h./norm_const;
        % Normalize new_Sigma_h
        new_Sigma_h= new_Sigma_h.*repmat(1/(norm_const.^2),L_h,L_h);
    end
end

end