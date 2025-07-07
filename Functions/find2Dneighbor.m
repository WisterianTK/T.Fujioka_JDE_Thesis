function index = find2Dneighbor(h,w,H,W)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find neighboring pixels of pixel at (h,w) in 4-connection lattice
%%% INPUT %%%
% (h,w): current pixel postion 
% H: Height
% W: Width
%%% OUTPUT %%%
% index: vector containing indices of neighboring pixels (in linear indexing of MATLAB)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Relative positions in Subscript indexing
relative_pos = [-1, 0;  % Top
                 0, 1;  % Right
                 1, 0;  % Bottom
                 0, -1];% Left
             
pos_h = h + relative_pos(:,1);
pos_w = w + relative_pos(:,2);

% Check if positions exceed the border
valid = (pos_h > 0 & pos_h <= H) & (pos_w > 0 & pos_w <= W);
pos_h = pos_h(valid);
pos_w = pos_w(valid);

% Convert to linear indices
index = sub2ind([H, W], pos_h, pos_w);
end
