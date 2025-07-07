function [Sigma_r, mu_r, Sigma_h, mu_h, Phi, Beta, Beta_dens, sigma_b, L_coef, ELBO, sigma_h] = ...
    RunJDE(num_iter, X, Y, Sigma_h, mu_h, Phi, P, L_coef, Beta, sigma_b, sigma_h, Sigma_K, R, u_lb, u_ub, threshold, decay_update, normalize, loc_r, loc_h)

L_h = length(mu_h);
L_r = size(X,2) + 1 - L_h;

circ_mask = MakeCircshiftMask(L_h,L_r);
elbo_diff = 1000;
% Run JPDE
ELBO = zeros(num_iter,1);
for i=1:num_iter
    % VE-r (update on r)
    [Sigma_r, mu_r] =...
        JdeUpdateVEA(Sigma_h, mu_h, X, Y, P, L_coef, sigma_b, Sigma_K, Phi, loc_r);


    % VE-H (update on h)
    [Sigma_h, mu_h] = JdeUpdateVEH(Sigma_r, mu_r, X, sigma_h, R, Y, P, L_coef, sigma_b, circ_mask, normalize, loc_h);

    % VE-Z (update on z) 
    Phi = JdeUpdateVEQ(Sigma_r, mu_r, X, Y, Sigma_K, Phi, Beta, 0.2, threshold, elbo_diff, loc_r); % True for hard thresholding

    % VE-Beta
    [Beta, Beta_dens] = JdeUpdateVEBeta_uniform(Phi, u_lb, u_ub);

    % VM-L
    L_coef = JdeUpdateVML(mu_h, mu_r, X, Y, P);

    % VM-sigma_b
    sigma_b = JdeUpdateVMsigma_b(Sigma_h, mu_h, Sigma_r, mu_r, X, Y, P, L_coef, circ_mask);

    % Update after 10 iterations
    if i > 10
        % VM-sigma_h
        sigma_h = JdeUpdateVMsigma_h(Sigma_h, mu_h, R, 1, loc_h);
    end


    % ELBO
    ELBO(i) = JdeELBO_fast(Sigma_h, mu_h, Sigma_r, mu_r, X, sigma_h, R, Y, P, L_coef, sigma_b, Sigma_K, Phi, Beta, Beta_dens, u_ub, u_lb, loc_r, loc_h);

    if i > 1
        elbo_diff = ELBO(i) - ELBO(i-1);
        if elbo_diff < 1
            break
    elseif isnan(elbo_diff)
            break
        end
    end

end

ELBO = ELBO(1:i);

end


