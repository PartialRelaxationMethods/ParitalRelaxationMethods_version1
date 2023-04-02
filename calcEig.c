/* This function tracks the eigenvalues of a Hermitian matrix perturbed by
 *a Hermitian rank-one matrix D + \rho*z*z^H. The algorithm used in this function is based
 *on the paper: "Rank-one Modification of the Symmetric Eigenproblem".
 * Input:
 *        d: a row vector of size 1 x nAnt: the initial eigenvalues without
 *           any perturbation, which is also the elements on the diagonal of
 *           the matrix D. Elements of d must be sorted ascendingly
 *        rho: a row vector of size 1 x nGrid: the scaling factor. Must be real
 *        absz2: a row vector of size 1 x (nAnt*nGrid): contains the absolute
 *               value squared of the vector z for each grid point. For example,
 *               the first NAnt-values of absz2 are the absolute value squared
 *               of the first gridpoint and so on...
 *        eigOrder: a row vector of size 1 x nEig, contains the order of the
 *                  eigenvalues that should be tracked. The values in eigOrder
 *                  must be between 1 and nAnt.
 *          nAnt: the size of the matrix D
 *          nGrid: the number of gridpoints (steering vectors) for tracking
 *Output:
 *          eigVal: a row vector of size 1 x (nEig*nGrid): returns the tracked
 *                  eigenvalues for each grid point. For example, the first nEig
 *                  values of eigVal are the eigenvalues of the first grid point
 */

#include "mex.h"
#include "math.h"
#include <mkl_lapack.h>
#include <mkl_cblas.h>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_service.h>
#include <omp.h>

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    /* input and output variables*/
    double *d;
    double *rho;
    double *absz;
    double *eigOrder;
	MKL_INT nAnt, index, info, nEig, nGrid;
    //mwSize nEig;
    //mwSize nGrid;
    double *eigVal;              /* output matrix */
    /*temporary variables in the routine*/
    
    MKL_INT iGrid;
	double eig;
    
    /* Initialize values for input and output variables*/
    d = mxGetPr(prhs[0]);
    rho = mxGetPr(prhs[1]);
    absz = mxGetPr(prhs[2]);
    eigOrder = mxGetPr(prhs[3]);
    nEig = (MKL_INT)mxGetN(prhs[3]);
    nAnt = (MKL_INT)mxGetScalar(prhs[4]);
    nGrid = (MKL_INT)mxGetScalar(prhs[5]);
    plhs[0] = mxCreateDoubleMatrix(nEig,nGrid,mxREAL);
    eigVal = mxGetPr(plhs[0]);
    
	MKL_INT *intEigOrder = (MKL_INT*) mkl_malloc(nEig * sizeof(MKL_INT), 64);
	for (MKL_INT iEig = 0; iEig < nEig; iEig++) {
		intEigOrder[iEig] = (MKL_INT)eigOrder[iEig];
	}
	//double *delta = (double*)malloc(nAnt * sizeof(double));
    double *delta = (double*) mkl_malloc(nAnt * sizeof(double), 64);
	for (iGrid = 0; iGrid < nGrid; iGrid++) {
        for (MKL_INT iEig = 0; iEig < nEig; iEig++) {
            dlaed4(&nAnt, intEigOrder + iEig, d, absz + iGrid*nAnt, delta, rho + iGrid, eigVal + (iGrid*nEig+iEig), &info);
        }
        
    }
    mkl_free(intEigOrder);
	mkl_free(delta);
    printf("PR method (mex-function)\n"); 
    printf("MIT License \nCopyright (c)"); 
    printf("2023 Trinh-Hoang, Minh \n");
    printf("Permission is hereby granted, free of charge, to any person obtaining a copy \n");
    printf("of this software and associated documentation files (the \'Software\'), to deal \n");
    printf("in the Software without restriction, including without limitation the rights \n");
    printf("to use, copy, modify, merge, publish, distribute, sublicense, and/or sell \n");
    printf("copies of the Software, and to permit persons to whom the Software is \n");
    printf("furnished to do so, subject to the following conditions: \n \n");
    printf("The above copyright notice and this permission notice shall be included in all\n");
    printf("copies or substantial portions of the Software. \n \n");
    printf("THE SOFTWARE IS PROVIDED \'AS IS\', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR \n");
    printf("IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, \n");
    printf("FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE \n");
    printf("AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER \n");
    printf("LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, \n");
    printf("OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE \n");
    printf("SOFTWARE. \n \n");
    printf("If you use this software in any publication, you must acknowledge the author\n");
    printf("and cite the corresponding paper. The citation for the paper is  \n \n");
    printf("M. Trinh-Hoang, M. Viberg, and M. Pesavento, \"Partial relaxation approach: \n");  
    printf("An eigenvalue-based DOA estimator framework\", IEEE Transactions on Signal \n");
    printf("Processing, vol. 66, no. 23, pp. 6190–6203, Dec. 2018, doi: 10.1109/TSP.2018.2875853.\n\n");
    }