// Need to compile this c++ code in MATLAB as below
// Eigenpath = ['-I' pwd '/Eigen'];
// Omppath = '-I/hoge/include'; % Change Dir
// ompcommand = ['-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'];
// mex(Eigenpath, Omppath, 'calcFA.cpp', ompcommand);

#include <stdio.h>
#include <string.h> /* strlen */
#include <math.h>
#include <iostream>
#include <vector>
#include "mex.h"
using namespace std;

#include <Eigen>
using namespace Eigen;

#include <omp.h>

float calFA(MatrixXf sVal) {
    const int size = sVal.rows();
    float sum1 = 0;
    float sum2 = 0;
    for (int i = 0; i < size; i++){
        sum1 += (sVal(i) - sVal.mean()) * (sVal(i) - sVal.mean());
        sum2 += sVal(i)*sVal(i);
    }
    sum2 += FLT_MIN;
    return sqrt( ( float(size)/float(size-1) ) * sum1 / sum2 );
}

// Main //
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{   
    // Get Input Information
    const mwSize* dimArray = mxGetDimensions(prhs[0]); // mwSize == int, mwSize is better in mexMatrix
    const mwSize nF = dimArray[0];
    const mwSize nH = dimArray[1];
    const mwSize nW = dimArray[2];
    const mwSize nO = dimArray[3];

    // Read Data
    float* pr = (float*)mxGetPr(prhs[0]);
    int prIDX = 0;    
    
    // set patameter
    const int tWindow = int(mxGetScalar(prhs[1])); // temporal window
    const int sWindow = int(mxGetScalar(prhs[2])); // spacial window
    int tmpIDX = 0;
   
    // variables for output 
    mwSize dims[3] = {nF, nH, nW};
    float* pl0;
    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    pl0 = (float*)mxGetPr(plhs[0]);
    int plIDX = 0;
    
    #pragma omp parallel for num_threads(omp_get_max_threads()) private(prIDX, tmpIDX, plIDX) 
    for(int f=0+tWindow/2; f<nF-tWindow/2; f++){
        for(int h=0; h<nH; h++){
            for(int w=0; w<nW; w++){
                int sizeh = 0;
                int sizew = 0;
                for(int i=0; i<sWindow+1; i++){
                    int hpos = h-sWindow/2+i;
                    int wpos = w-sWindow/2+i;
                    if( hpos >=0 && hpos < nH ){ sizeh++; }
                    if( wpos >=0 && wpos < nW ){ sizew++; }
                }

                MatrixXf matrix = MatrixXf::Zero(sizeh*sizew*(tWindow+1), nO);               

                for(int o=0; o<nO; o++){
                    tmpIDX = 0;
                    for(int fw=f-tWindow/2; fw<f+tWindow/2+1; fw++){
                        for(int hw=h-sWindow/2; hw<h+sWindow/2+1; hw++){
                            for(int ww=w-sWindow/2; ww<w+sWindow/2+1; ww++){                                    
                                if( hw>=0 && hw<nH && ww>=0 && ww<nW){
                                    prIDX = fw + hw * nF + ww * (nF * nH) + o * (nF * nH * nW);
                                    matrix(tmpIDX,o) = pr[prIDX];
                                    tmpIDX++;
                                }                                  
                            }
                        }
                    }
                }

                // Zero mean
                matrix = matrix.rowwise() - matrix.colwise().mean();
                
                // SVD
                JacobiSVD<MatrixXf> svd(matrix);
                
                // Eigenvalues
                MatrixXf lambda = svd.singularValues().cwiseProduct(svd.singularValues()) / float(sizeh*sizew*(tWindow+1) - 1);
                
                // output data         
                plIDX = f + h * nF + w * (nF * nH);
                pl0[plIDX] = calFA(lambda); // FA
            }
        }   
    }
}