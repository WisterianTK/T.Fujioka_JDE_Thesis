function new_Phi =...
    JdeUpdateVEQ(Sigma_r, mu_r, X, Y, Sigma_K, Phi, Beta, threshold, option, elbo_diff, loc_r)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update for posterior of neural activation states
%%% INPUT %%%
% Sigma_r: Current estimate of covariances of NRFs over EPs [H x W x (L_r*M) x (L_r*M)]  (L_r:length of NRF, M:number of EPs)
% mu_r: Current estimate of mean of NRFs        [H x W x (L_r*M)]
% X: Design(Toeplitz) matrices based on EPs [N x (L_h+L_r-1) x M] (N:number of time samples)
% Y: fUS data                              [H x W x N]
% Sigma_K: Covariance matrices for each activation class and EP [L_r x L_r x M x K]
% Phi: Mixture assignment of neural activation (Categorical dist.)  [H x W x M x K] (K:number of activation classes)
% (i.e., Phi(h,w,m,k) is a probability of pixel (h,w) being in state k for mth condition)
% (i.e., sum(Phi,4) = ones(H,W,M))
% Beta: Spatial regulation parameter in Potts/Ising model       [M x 1]
% loc_r: prior mean of NRF [L_r x M x K]

%%% OUTPUT %%%
% new_Phi: New estimate of soft mixture assignment weights [H x W x M x K] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(Y,1);    % Height
W = size(Y,2);    % Width
M = size(X,3);    % Number of EPs
K = size(Phi,4);  % Number of activation classes
L_r = size(Sigma_K,1);  % Filter length of NRF

new_Phi = Phi;   % Initialize new estimate for Phi

% Update Phi_0 and Phi_1
for h = 1:H
    for w = 1:W
        for m = 1:M
            mu_r_jm = squeeze(mu_r(h,w,(L_r*(m-1)+1):(L_r*m))); % Extract mu_r for jth pixel under mth condition
            Sigma_r_jm = squeeze(Sigma_r(h,w,(L_r*(m-1)+1):(L_r*m),(L_r*(m-1)+1):(L_r*m))); % Extract covariance for jth pixel under mth condition
            for k = 1:K
                loc_mk = loc_r(:,m,k);
                % Find neighbors of current pixel at (h,w)
                neighbor_index = find2Dneighbor(h,w,H,W);
    
                % Compute sum of mixture assignment weights of all neighbors
                % for state k and mth condition . H*W*(m-1) is for indexing of EPs
                new_Phi_k = squeeze(new_Phi(:,:,:,k));
                neighborSum_km = sum(new_Phi_k(neighbor_index + H*W*(m-1)));

            %     % Evaluate multivariate normal distribution at mu_r_jm
            %     multi_normal = 1/((2*pi)^(L_r/2)*det(Sigma_K(:,:,m,k))^(1/2))*...
            %     exp(-0.5*mu_r_jm'/(Sigma_K(:,:,m,k))*mu_r_jm);
            % 
            %     % Compute unnormalized posterior probability
            %     new_Phi(h,w,m,k) = multi_normal*exp(-0.5*trace(Sigma_K(:,:,m,k)\Sigma_r_jm) + Beta(m)*neighborSum_km);
            % end
            % 
            % % Normalize the mixture weights
            % norm_const = sum(new_Phi(h,w,m,:));
            % if norm_const==0
            %     new_Phi(h,w,m,:) = Phi(h,w,m,:);
            % else
            %     new_Phi(h,w,m,:) = new_Phi(h,w,m,:)/norm_const;
            % end

                % Evaluate multivariate normal distribution at mu_r_jm in log form (to avoid underflow)
                log_multi_normal = log(1/((2*pi)^(L_r/2)*det(Sigma_K(:,:,m,k))^(1/2)))...
                -0.5*(mu_r_jm-loc_mk)'/(Sigma_K(:,:,m,k))*(mu_r_jm-loc_mk);

                % Compute unnormalized log posterior probability
                new_Phi(h,w,m,k) = log_multi_normal - 0.5*trace(Sigma_K(:,:,m,k)\Sigma_r_jm) + Beta(m)*neighborSum_km;

            end
            % Use LogSumExp trick to avoid underflow
            log_Z = logsumexp(squeeze(new_Phi(h,w,m,:)));  % log normalization factor
            log_p_normalized = squeeze(new_Phi(h,w,m,:))- log_Z;
            new_Phi(h,w,m,:) = exp(log_p_normalized);

            % new_Phi(h,w,m,:) = 0.8*squeeze(Phi(h,w,m,:)) + 0.2*exp(log_p_normalized);

            % % Avoid 0 
            % if ~all(new_Phi(h,w,m,:)-1)
            %     new_Phi(h,w,m,:) = (new_Phi(h,w,m,:)+0.01*sum(new_Phi(h,w,m,:),"all"))/sum(new_Phi(h,w,m,:)+0.01*sum(new_Phi(h,w,m,:),"all"));
            % end
            % % Hard threshold 
            % if option && elbo_diff<100 && (max(abs(mu_r_jm)) < threshold)
            %     new_Phi(h,w,m,1) = 1e-1;
            %     % Normalize
            %     new_Phi(h,w,m,:) = new_Phi(h,w,m,:)/sum(new_Phi(h,w,m,:),"all");
            % end

        end
    end
end

end


