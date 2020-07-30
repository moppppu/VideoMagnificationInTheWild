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
    sum2 += FLT_MIN; // 0�Ŋ���̂�h��
    return sqrt( ( float(size)/float(size-1) ) * sum1 / sum2 );
}

// Main //
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{   
    // Get Input Information
    const int* dimArray = mxGetDimensions(prhs[0]);
    const int nF = dimArray[0];
    const int nH = dimArray[1];
    const int nW = dimArray[2];

    // Read Data
    float* pr = (float*)mxGetPr(prhs[0]);  // ����1�����Ƃ��ēǂݍ��ނ���C��t����
    int prIDX = 0;    
    
    // ADC & FA Calulation
    const int tWindow = int(mxGetScalar(prhs[1])); // temporal window
    const int sWindow = int(mxGetScalar(prhs[2])); // spacial window
    int tmpIDX = 0;
    int smpIDX = 0;
   
    // variables for output 
    mwSize dims[3] = {nF, nH, nW};
    float* pl0; // mxCreateNumericArray������ɐ錾����K�v������i�Ȃ����͕s���j
    float* pl1;  
    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); // output�̔z��p��
    plhs[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    pl0 = (float*)mxGetPr(plhs[0]); // output�z��̃|�C���^��GET����
    pl1 = (float*)mxGetPr(plhs[1]);
    int plIDX = 0;
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // parallel����for�I�ȐU�镑���i�C���f�b�N�X�j������ϐ��̓}���`�X���b�h�œƗ��ɐ錾���������̂ŁCprivate���s���K�v������
    // �Ƃ������Cparallel�O�Ő錾���ꂽ�ϐ��́C�S�āh���ʁh�Ŏg�p����邽�ߒ��ӂ���D����parallel���Œl���ς����̂͌ʂɐ錾���Ȃ��ƃ_���I
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
//     printf("�g�p����X���b�h���́u%d�i�ő�j�v�ł��B\n", omp_get_max_threads() );
    
    #pragma omp parallel for num_threads(omp_get_max_threads()) private(prIDX, tmpIDX, smpIDX, plIDX) 
    for(int f=0+tWindow/2; f<nF-tWindow/2; f++){
        for(int h=0; h<nH; h++){
            for(int w=0; w<nW; w++){
                
                //�@2D���߂�
                int sizeh = 0;
                int sizew = 0;
                for(int i=0; i<sWindow+1; i++){
                    int hpos = h-sWindow/2+i;
                    int wpos = w-sWindow/2+i;
                    if( hpos >=0 && hpos < nH ){ sizeh++; }
                    if( wpos >=0 && wpos < nW ){ sizew++; }
                }

                MatrixXf matrix = MatrixXf::Zero( (tWindow+1), (sizeh*sizew) );
                
                tmpIDX = 0;
                for(int fw=f-tWindow/2; fw<f+tWindow/2+1; fw++){
                    smpIDX = 0;
                    for(int hw=h-sWindow/2; hw<h+sWindow/2+1; hw++){
                        for(int ww=w-sWindow/2; ww<w+sWindow/2+1; ww++){                                    
                            if( hw>=0 && hw<nH && ww>=0 && ww<nW){
                                prIDX = fw + hw * nF + ww * (nF * nH); // �ЂƂO�̃T�C�Y�������V�t�g������K�v������      
                                matrix(tmpIDX,smpIDX) = pr[prIDX];
                                smpIDX++;
                            }
                        }
                    }
                    tmpIDX++;
                }
                
                // 0����̕ω��ɂ���i�������ɕ��ς������j
                matrix = matrix.rowwise() - matrix.colwise().mean();
                
                // SVD�i�v�Z�d���j
                JacobiSVD<MatrixXf> svd(matrix); // ���ْl�����~�����Ȃ炱��CComputeFullU������U�{�CComputeThinU������Q�{����
                
                // ���ْl�������U�s��̌ŗL�l�iPCA�j�ɕϊ�����
                MatrixXf lambda = svd.singularValues().cwiseProduct(svd.singularValues()) / float(tWindow+1 - 1);
                
                // output data         
                plIDX = f + h * nF + w * (nF * nH); // �ЂƂO�̃T�C�Y�������V�t�g������K�v������    
                pl0[plIDX] = lambda.mean(); // ADC
                pl1[plIDX] = calFA(lambda); // FA
            }
        }   
    }
    
}