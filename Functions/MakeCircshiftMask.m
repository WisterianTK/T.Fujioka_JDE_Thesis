function circ_mask = MakeCircshiftMask(L_h, L_r)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate mask for a tensor containing circshift, used in VEH and VM_sigma_b
%%% INPUT %%%
% L_h: Length of HRF
% L_r: Length of NRF
%%% OUTPUT %%%
% mask: Tensor of size [(L_r + L_h - 1) x (L_r + L_h - 1) x L_h x L_h]
% (i.e., mask(:,:,a,b) contains block matrix ones(L_r,L_r) shifted by a-1, b-1 in row and column directions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toeplength = L_r+L_h-1;

mask = zeros(toeplength);
mask(1:L_r,1:L_r) = 1;
circ_mask = zeros(toeplength,toeplength,L_h,L_h);
for a = 1:L_h
    for b=1:L_h
        circ_mask(:,:,a,b) = circshift(mask,[b-1,a-1]);
    end
end


% B = zeros(toeplength,toeplength,L_h,L_h);
% B(boolean(mask_list)) = repmat(A,1,L_h,L_h);

end



