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
function [ DOA_est, Spec_est ] = prwsf( Y, NSrc, Grid, AGrid, varargin )
%PRDML returns the estimated DOA from the received signal and the
%estimated spectrum if requested using the proposed algorithm of Mats
%Viberg, which is based on the partially-relaxed WSF with the same
%weighting as WSF. The method of determining the eigenvalues is using
%rational approximation. For more information, please refer to 
%"Solving Secular Function Stably and Efficiently"
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

% Perform full eigenvalue decomposition and diagonalization
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

% Calculate the factor rho of the rank-one term for each angle of view
rho = 1/NAnt;

% Calculate the absolute value squared of each entry in the z vector for
% each angle of view
% ATilde = Us'*AGrid;
% Z = bsxfun(@times, ATilde, sqrt(w));
% absZ = abs(Z);

absZ = abs(sqrt(w).*(Us'*AGrid));
wM_spec = sum(absZ.^2, 1);


% Calculate the N-th modified eigenvalue required in the paper for each angle of view
% Note that here we use the rational approximation implemented in mexC.
neg_eigval = calcEig(-w, rho*wM_spec, absZ./sqrt(wM_spec), NSrc, NSrc, NGrid);

% Calculate and normalize the pseudo-spectrum
Spec_est = 1./(-neg_eigval + 1e-6);
Spec_est = Spec_est./max(abs(Spec_est));

% Estimate the DOA based on the pseudo-spectrum
[~, locs] = findpeaks(Spec_est, 'SortStr', 'descend');
if length(locs)> NSrc
    DOA_est = Grid(locs(1:NSrc));
else
    DOA_est = [Grid(locs), 90*ones(1, NSrc - length(locs))];
end
DOA_est = sort(DOA_est(:));
end