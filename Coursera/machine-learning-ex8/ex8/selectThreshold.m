function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

% for testing ;-)
% yval = [ 1 ; 0 ; 1 ; 0 ; 1];
% pval = [.11 ;.22 ;.32 ;.62 ;.82];
% stepsize = (max(pval) - min(pval)) / 3

stepsize = (max(pval) - min(pval)) / 1000;

% ditch the loop, we do not need it.
% for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

ep = [min(pval):stepsize:max(pval)];


% y = 1 if p < e  
% y = 0 if p >= e
y = bsxfun(@lt,pval,ep); 

% predicted is 1 , actual 1 count 
tp = sum(y.*yval);  


% flase -ve : actual (yval) 1, predicted (y) is 0 
% hence y .* yval
% bsxfun(@gt, yval, y .* yval)
fn = sum(bsxfun(@gt, yval, y .* yval));
 
% flase +ve : actual (yval) 0, predicted (y) is 1 
% bsxfun(@lt, yval, y .* (1-yval))
fp = sum(bsxfun(@lt, yval, y .* (1-yval)));


prec = tp ./(tp+fp);

rec = tp  ./(tp+fn);

F1 =  2 * (prec .* rec) ./ (prec + rec);

[bestF1, idx]    = max(F1, [], 2);

bestEpsilon = ep(idx);


% =============================================================
% ditch the loop
%     if F1 > bestF1
%        bestF1 = F1;
%        bestEpsilon = epsilon;
%     end
% end

end
