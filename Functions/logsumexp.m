function lse = logsumexp(log_vals)
    max_log = max(log_vals);
    lse = max_log + log(sum(exp(log_vals - max_log)));
end