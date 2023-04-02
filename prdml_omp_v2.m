% MIT License
% 
% Copyright (c) 2023 Trinh-Hoang, Minh
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
% 
% If you use this software in any publication, you must acknowledge the author 
% and cite the corresponding paper. The citation for the paper is 
% 
% M. Trinh-Hoang, M. Viberg, and M. Pesavento, “Partial relaxation approach: 
% An eigenvalue-based DOA estimator framework,” IEEE Transactions on Signal 
% Processing, vol. 66, no. 23, pp. 6190–6203, Dec. 2018, doi: 10.1109/TSP.2018.2875853.
% 
function [ DOA_est] = prdml_omp_v2( Y, NSrc, Grid, AGrid, varargin )
%PRWSF returns the estimated DOA from the received signal using the OMP
%principle together with the PR-WSF spectral search in each iteration
%
%Input:   Y: Received signal (NAnt x NSnp)
%         NSrc: Number of sources
%         Grid: the sampled angle of view (1 x NGrid)
%         AGrid: the dictionary steering matrix corresponding to the
%         sampled angle of view (NAnt x NGrid)
%         Ryy: the covariance matrix of the received signal (optional)
%
% Output: DOA_est: the estimated DOA on the sampled angle of view (1 x NSrc)
%         Spec_est: the normalized pseudo-spectrum on the grid,  (1 x NGrid)
%                   where the highest peak has the unit height

% Initialize parameters
[NAnt, NSnp] = size(Y);
NGrid = length(Grid);
if isempty(varargin)
    R_est = 1/NSnp*(Y*Y');
else
    R_est = varargin{1};
end

[U, Lambda_hat] = eig(R_est);
lambda_hat = diag(real(Lambda_hat));
[lambda_hat, order] = sort(lambda_hat, 'descend');
U = U(:, order);

% Calculate the elements on the diagonal of the weighting matrix W
w = lambda_hat;
neg_w = -w;
V_mat = complex(diag(sqrt(w)));
ADict = AGrid./sqrt(sum(abs(AGrid).^2, 1));
ADictTransform = U'*ADict;
[DOA_est, ~] = calcDOA_PRDML_OMP_v2(neg_w, V_mat, Grid, ADictTransform, NSrc);
% 
% for (iFig = 1:NSrc)
%    figure(iFig);
%    [~, ~, ind] = intersect(DOA_est(1:iFig-1), Grid);
%    objVal(ind, iFig) = inf;
%    hold on
%    plot(-objVal(:, iFig));
% end
DOA_est = sort(DOA_est(:));

end