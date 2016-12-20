function [A,B] = highdim_pca(X,T,d,embeddedlags,standardise)
% pca for potentially loads of subjects
%
% if X is a cell of things, uses SVD
% if X is a matrix, uses Matlab's PCA 
%
% d indicates how many PCA components to take, and can be specified 
%   in different ways:
% if length(d)==1 and d is lower than 1, then that's the proportion of
%   variance to keep
% if length(d)==1 and d is higher than 1, this is the number of components
% if d is a vector (d1,d2), then d1 must be <1 and d2 must be >=1 ;
%   in this case it will take the minimum of d2 and the number of
%   components that explain d1 amount of variance. 
% if d is a vector (d1,d2,d3), is the same than before, but d3 indicates
%   whether to take the minimum (d3=1) or the maximum (d3=2)
%
% Author: Diego Vidaurre, University of Oxford (2016)

if nargin<3, embeddedlags = 0; end
if nargin<4, standardise = 1; end

is_cell_strings = iscell(X) && ischar(X{1});
is_cell_matrices = iscell(X) && ~ischar(X{1});
options = struct();
options.standardise = standardise;
options.embeddedlags = embeddedlags;
options.pca = 0; % PCA is done here! 

if is_cell_strings || is_cell_matrices
    B = [];
    for i=1:length(X)
        X_i = loadfile(X{i},T{i},options); % embedded is done here
        X_i = X_i - repmat(mean(X_i),size(X_i,1),1); % must center
        if i==1, C = zeros(size(X_i,2)); end
        C = C + X_i' * X_i;
    end
    [A,e,~] = svd(C); 
    e = diag(e);  
else
    if length(embeddedlags)>1
       X = embeddata(X,T,options.embeddedlags); 
    end
    [A,B,e] = pca(X,'Centered',true);     
end
e = cumsum(e)/sum(e);

if length(d)==1 && d<1
    ncomp = find(e>d,1);
elseif length(d)==1 && d>=1
    ncomp = d;
elseif length(d)==2 || (length(d)==3 && d(3)==1)
    ncomp = min(find(e>d(1),1),d(2));
elseif length(d)==3 && d(3)==2
    ncomp = max(find(e>d(1),1),d(2));
else
    error('pca parameters are wrongly specified')
end

fprintf('Working in PCA space, with %d components... \n',ncomp)

A = A(:,1:ncomp);
if ~isempty(B), B = B(:,1:ncomp); end

end