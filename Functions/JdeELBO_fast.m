function elbo =...
    JdeELBO_fast(Sigma_h, mu_h, Sigma_r, mu_r, X, sigma_h, R, Y, P, L_coef, sigma_b, Sigma_K, Phi, Beta, Beta_dens, u_ub, u_lb, loc_r, loc_h)

H = size(Y,1);    % Height
W = size(Y,2);    % Width
M = size(X,3);    % Number of EPs
N = size(Y,3);    % Number of time samples
K = size(Phi,4);
L_r = size(mu_r,3)/M;
L_h = size(mu_h,1);

log_conditional = 0; % E[ln( p(Y|H,R) )]
cross_entropy1 = 0;  % E[ln( p(R|Q) )]
cross_entropy2 = 0;  % E[ln( p(Q|Beta) )]
entropy2 = 0;        % E[ln( q(r) )]

Y_vec = reshape(Y,[],N);
L_vec = reshape(L_coef,[], size(L_coef,3));
mu_r_vec = reshape(mu_r, [], size(mu_r,3));
Sigma_r_vec = reshape(Sigma_r,[],size(Sigma_r,3),size(Sigma_r,3));
Phi_vec = reshape(Phi,[],M,K);

% Precompute determinant of \Sigma_K
det_Sigma_K = zeros(M,K);
for m=1:M
    for k=1:K
        det_Sigma_K(m,k) = det(Sigma_K(:,:,m,k));
    end
end

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


for j = 1:H*W
    y_j = Y_vec(j,:)';
    el_j = L_vec(j,:)';
    mu_rj = mu_r_vec(j,:)';
    Sigma_rj = squeeze(Sigma_r_vec(j,:,:));
    % log_conditional
    log_conditional = log_conditional - 1/2*(N*log(2*pi) + N*log(sigma_b(j)) ...
        + 1/sigma_b(j)*((y_j - P*el_j)'*(y_j - 2*exG*mu_rj - P*el_j)...
        + mu_rj'*exGG*mu_rj  + trace(exGG*Sigma_rj)));

    % Find neighbors of current pixel at (h,w)
    [h,w] = ind2sub([H,W],j);
    neighbor_index = find2Dneighbor(h,w,H,W);
    

    for m=1:M
        mu_rjm = mu_rj((m-1)*L_r+1:m*L_r);
        Sigma_rjm = Sigma_rj((m-1)*L_r+1:m*L_r, (m-1)*L_r+1:m*L_r);
        pseudo_log_normalization = 0;
        for k=1:K
            Phi_jmk = Phi_vec(j,m,k);
            Sigma_km = Sigma_K(:,:,m,k);
            loc_r_mk = loc_r(:,m,k);
            % Cross_entropy1
            % log_det_Sigma_K_mk = 2 * sum(log(diag(chol(det_Sigma_K(m,k)))));
            cross_entropy1 = cross_entropy1 - Phi_jmk/2*(L_r*log(2*pi)...
                + log(det_Sigma_K(m,k)) + (mu_rjm - loc_r_mk)'*(Sigma_km\(mu_rjm - loc_r_mk)) ...
                + trace(Sigma_km\Sigma_rjm));
            % cross_entropy1 = cross_entropy1 - Phi_jmk/2*(L_r*log(2*pi)...
            %     + log_det_Sigma_K_mk + (mu_rjm - 2*loc_r_mk)'*Sigma_km*mu_rjm ...
            %     + loc_r_mk'*Sigma_km*loc_r_mk + trace(Sigma_km\Sigma_rjm));

            neighborSum_km = sum(Phi_vec(neighbor_index,m,k));
            % Cross_entropy2
            cross_entropy2 = cross_entropy2 + Beta(m)*neighborSum_km;
            pseudo_log_normalization = pseudo_log_normalization + exp(Beta(m)*neighborSum_km);
        end
        % Cross_entropy2
        cross_entropy2 = cross_entropy2 - log(pseudo_log_normalization);     

        % Entropy2
        log_det_Sigma_rjm = 2 * sum(log(diag(chol(Sigma_rjm))));
        % entropy2 = entropy2 + 1/2*(L_r*log(2*pi) + L_r + log(det(Sigma_rjm)));
        entropy2 = entropy2 + 1/2*(L_r*log(2*pi) + L_r + log_det_Sigma_rjm);
    end

end

% Cross_entropy3      % E[ln( p(Beta) )]
cross_entropy3 = -sum(Beta_dens*log(u_ub - u_lb), "all");

% Cross_entropy4      % E[ln( p(H) )]
log_det_sigma_hR = 2 * sum(log(diag(chol(sigma_h*R))));

% cross_entropy4 = -1/2*(L_h*log(2*pi) + log(det(sigma_h*R)) ...
%     + mu_h'/(sigma_h*R)*mu_h + trace((sigma_h*R)\Sigma_h));
cross_entropy4 = -1/2*(L_h*log(2*pi) + log_det_sigma_hR ...
    + (mu_h-loc_h)'/(sigma_h*R)*(mu_h-loc_h) + trace((sigma_h*R)\Sigma_h));

log_det_Sigma_h = 2 * sum(log(diag(chol(Sigma_h))));
% entropy1 = 1/2*(L_h*log(2*pi) + L_h + log(det(Sigma_h))); % Entropy of q(h)
entropy1 = 1/2*(L_h*log(2*pi) + L_h + log_det_Sigma_h); % Entropy of q(h)
entropy3 = -sum(log(Phi.^Phi),"all");                      % Entropy of q(Q)
entropy4 = -sum(log(Beta_dens.^Beta_dens),"all");   % Entropy of q(Beta)

elbo = log_conditional + cross_entropy1 + cross_entropy2 + cross_entropy3...
    + cross_entropy4 + entropy1 + entropy2 + entropy3 + entropy4;
end

    











