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
function [ DOA_est] = prwsf_omp( Y, NSrc, Grid, AGrid, varargin )
%PRWSF returns the estimated DOA from the received signal and the
%estimated spectrum if requested using the proposed algorithm of Mats
%Viberg, which is based on the partially-relaxed DML. The
%method of determining the eigenvalues is using rational approximation. For
%more information, please refer to "Solving Secular Function Stably and 
%Efficiently"
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
Us = U(:, 1:NSrc);
lambda_s = lambda_hat(1:NSrc);
lambda_n = lambda_hat(NSrc+1:end);
sigma_n_squared_hat = mean(lambda_n);
w = (lambda_s - sigma_n_squared_hat*ones(NSrc,1)).^2./lambda_s;

R_mod = Us*diag(w)*Us';

proj = @(x) x*pinv(x);
oproj = @(x) eye(NAnt) - proj(x);
objVal = zeros(NGrid, 1);
indArr = zeros(NSrc, 1);
for (iSrc = 1:NSrc)
    for (iGrid = 1:NGrid)
        matA = [AGrid(:, indArr(1:iSrc-1)), AGrid(:, iGrid)];
%         a = AGrid(:, iGrid);
        eigVal = sort(real(eig(oproj(matA)*R_mod)), 'descend');
        objVal(iGrid) = sum(eigVal(NSrc - iSrc + 1:NSrc));
    end
%    figure(iSrc)
%    hold on
%    plot(Grid, objVal);
    [~, ind] = min(objVal);
    indArr(iSrc) = ind;
end
DOA_est = Grid(indArr);
DOA_est = sort(DOA_est(:));

end