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
clear all;
close all; 
%rng default;
%% Define simulation setup 
% number of antennas
NAnt = 13;
% Direction of Arrivals in degrees
DOA = [5;10];
% Number of sources
NSrc = length(DOA);
% Steering matrix
aDeg = @(DOA)exp(1j*pi*(0:NAnt-1).'*sin(DOA(:).'/180*pi));
A = aDeg(DOA);
% Number of snapshots
NSnp = 100;
% SNR at the receiver in dB
SNRdB = 10;% 0;
%% Signal Model
% Transmitted signal
S = 1/sqrt(2)*(randn(NSrc, NSnp) + 1j*randn(NSrc, NSnp));
Rss = eye(NSrc);
% Noise at the receiver
N = 10^(-SNRdB/20)/sqrt(2)*(randn(NAnt, NSnp) + 1j*randn(NAnt, NSnp));
% Received signal
Y = A*sqrtm(Rss)*S + N;
% Grid points of the spectrum and the corresponding steering dictionary
NGrid = 3500;
Grid = linspace(-90, 90, NGrid);
AGrid = aDeg(Grid);
% -------------------------------------------------------------------------
% Partial Relaxation Determininstic Maximum Likelihood (PR DML) Algorithm
% -------------------------------------------------------------------------
new_func2 = @prdml;
tic
[estDOA_PR_DML, estSpec_PR_DML] = new_func2(Y, NSrc, Grid, AGrid);
ExecutionTime_PR_DML=toc;
figure
hold on
plot(Grid, estSpec_PR_DML);
plot([DOA(1) DOA(2);DOA(1) DOA(2)], [0 0;1 1], '-.k');
axis([-90 90 0 1]);
title('Pseudo-spectrum PR-DML')
legend('PR-DML','True DOAs');
xlabel('Angle (deg)')
ylabel('Normalized pseudo-spectrum')
fprintf('PR-DML estimation:\n');
fprintf('Elapsed time is %f seconds.\n',ExecutionTime_PR_DML);
fprintf('Estimate source 1: %2.5f (deg); Estimate source 2: %2.5f (deg)\n\n',estDOA_PR_DML)
% -------------------------------------------------------------------------
% Partial Relaxation Constrained Covariance Fitting (PR CCF) Algorithm
% -------------------------------------------------------------------------
new_func2 = @prccf;
tic
[estDOA_PR_CCF, estSpec_PR_CCF] = new_func2(Y, NSrc, Grid, AGrid);
ExecutionTime_PR_CCF=toc;
figure
hold on
plot(Grid, estSpec_PR_CCF);
plot([DOA(1) DOA(2);DOA(1) DOA(2)], [0 0;1 1], '-.k');
axis([-90 90 0 1]);
title('Pseudo-spectrum PR-CCF')
legend('PR-CCF','True DOAs');
xlabel('Angle (deg)')
ylabel('Normalized pseudo-spectrum')
fprintf('PR-CCF estimation:\n');
fprintf('Elapsed time is %f seconds.\n',ExecutionTime_PR_CCF);
fprintf('Estimate source 1: %2.5f (deg); Estimate source 2: %2.5f (deg)\n\n',estDOA_PR_CCF)
% -------------------------------------------------------------------------
% Partial Relaxation Weighted Subspace Fitting (PR WSF) Algorithm
% -------------------------------------------------------------------------
new_func2 = @prwsf;
tic
[estDOA_PR_WSF, estSpec_PR_WSF] = new_func2(Y, NSrc, Grid, AGrid);
ExecutionTime_PR_WSF=toc;
figure
hold on
plot(Grid, estSpec_PR_WSF);
plot([DOA(1) DOA(2);DOA(1) DOA(2)], [0 0;1 1], '-.k');
axis([-90 90 0 1]);
title('Pseudo-spectrum PR-WSF')
legend('PR-WSF','True DOAs');
xlabel('Angle (deg)')
ylabel('Normalized pseudo-spectrum')
fprintf('PR-WSF estimation:\n');
fprintf('Elapsed time is %f seconds.\n',ExecutionTime_PR_WSF);
fprintf('Estimate source 1: %2.5f (deg); Estimate source 2: %2.5f (deg)\n\n',estDOA_PR_WSF)
% -------------------------------------------------------------------------
% Partial Relaxation Deterministic Maximum Likelihood Orthogonal Least Square (PR DML OLS) Algorithm
% Fast implementation using mex function
% -------------------------------------------------------------------------
new_func2 = @prdml_omp_v2;
tic
[estDOA_PR_DML_OLSmex] = new_func2(Y, NSrc, Grid, AGrid);
ExecutionTime_PR_DML_OLSmex=toc;
fprintf('PR-DML OLS estimation (accelerated version with mex-function):\n');
fprintf('Elapsed time is %f seconds.\n',ExecutionTime_PR_DML_OLSmex);
fprintf('Estimate source 1: %2.5f (deg); Estimate source 2: %2.5f (deg)\n\n',estDOA_PR_DML_OLSmex)
% -------------------------------------------------------------------------
% Partial Relaxation Deterministic Maximum Likelihood Orthogonal Least Square (PR DML OLS) Algorithm
% Fast implementation using mex function
% -------------------------------------------------------------------------
new_func2 = @prdml_omp;
tic
[estDOA_PR_DML_OLSmatlab] = new_func2(Y, NSrc, Grid, AGrid);
ExecutionTime_PR_DML_OLSmatlab=toc;
fprintf('PR-DML OLS estimation (Matlab version):\n');
fprintf('Elapsed time is %f seconds.\n',ExecutionTime_PR_DML_OLSmatlab);
fprintf('Estimate source 1: %2.5f (deg); Estimate source 2: %2.5f (deg)\n\n',estDOA_PR_DML_OLSmatlab)
% -------------------------------------------------------------------------
% Partial Relaxation Weighted Subspace Fitting Orthogonal Least Square (PR WSF OLS) Algorithm
% Fast implementation using mex function
% -------------------------------------------------------------------------
new_func2 = @prwsf_omp_v7;
tic
[estDOA_PR_WSF_OLSmex] = new_func2(Y, NSrc, Grid, AGrid);
ExecutionTime_PR_WSF_OLSmex=toc;
fprintf('PR-WSF OLS estimation (accelerated version with mex-function):\n');
fprintf('Elapsed time is %f seconds.\n',ExecutionTime_PR_WSF_OLSmex);
fprintf('Estimate source 1: %2.5f (deg); Estimate source 2: %2.5f (deg)\n\n',estDOA_PR_WSF_OLSmex)
% -------------------------------------------------------------------------
% Partial Relaxation Deterministic Maximum Likelihood Orthogonal Least Square (PR DML OLS) Algorithm
% Fast implementation using mex function
% -------------------------------------------------------------------------
new_func2 = @prwsf_omp;
tic
[estDOA_PR_WSF_OLSmatlab] = new_func2(Y, NSrc, Grid, AGrid);
ExecutionTime_PR_WSF_OLSmatlab=toc;
fprintf('PR-WSF OLS estimation (Matlab version):\n');
fprintf('Elapsed time is %f seconds.\n',ExecutionTime_PR_WSF_OLSmatlab);
fprintf('Estimate source 1: %2.5f (deg); Estimate source 2: %2.5f (deg)\n\n',estDOA_PR_WSF_OLSmatlab)

