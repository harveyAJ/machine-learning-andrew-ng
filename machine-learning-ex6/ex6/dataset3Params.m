function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

sigma_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
C_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
errors = zeros(3,1);

for i = 1:length(sigma_arr)
  for j=1:length(C_arr)
    currentSigma = sigma_arr(i);
    currentC = C_arr(j);
    fprintf(['Case sigma = %f C = %f :\n'], currentSigma, currentC);
   
    model= svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
    pred = svmPredict(model, Xval);
    currentError = mean(double(pred == yval)) * 100;
    errors = [errors [currentSigma; currentC; currentError]];
 
    fprintf(['Percentage of error for sigma = %f and C = %f: %f percent\n'], ...
         currentSigma, currentC, currentError);
     
  endfor
endfor

[min_error, index] = max(errors(3,:)); %min error is max accuracy
sigma = errors(1,index);
C = errors(2,index);

% =========================================================================

end
