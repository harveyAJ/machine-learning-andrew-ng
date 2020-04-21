function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

n = size(X, 2);

% You need to set these values correctly
X_norm = X;
mu = zeros(1, n);
sigma = zeros(1, n);     

for j=1:n 
  x = X(:,j);
  mu(j) = mean(x);
  sigma(j) = std(x);
  
  if (sigma(j) ~= 0)
    X_norm(:, j) = (x - mu(j)) / sigma(j);
  endif 
endfor

% ============================================================

end
