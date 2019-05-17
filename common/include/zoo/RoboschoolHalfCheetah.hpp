
#ifndef ROBOSCHOOLHALFCHEETAH_HPP
#define ROBOSCHOOLHALFCHEETAH_HPP

#include "bib/Logger.hpp"
#include "caffe/util/math_functions.hpp"
namespace zoo {
    
class RoboschoolHalfCheetah {

    static double weights_dense1_w [26*128];
    static double weights_dense1_b[128];
    static double weights_dense2_w[128*64];
    static double weights_dense2_b[64];
    static double weights_final_w[64*6];
    static double weights_final_b[6];
    
public:
    double* compute(const double* sensors) {
        double* y = new double[128]{0};
        
//        caffe::caffe_cpu_gemv(CblasNoTrans, 128, 26, (double)1.f, weights_dense1_w, sensors, (double)0., y);
        caffe_cpu_gemv(CblasNoTrans, 128, 26, (double)1.f, weights_dense1_w, sensors, (double)0., y);
        caffe::caffe_cpu_axpby(128, (double)1.f, weights_dense1_b, (double)1.f, y);
        
        //relu
        for (int i = 0; i < 128; ++i)
            y[i] = std::max(y[i], double(0));
        
        double* y2 = new double[64]{0};
        caffe_cpu_gemv(CblasNoTrans, 64, 128, (double)1.f, weights_dense2_w, y, (double)0., y2);
        caffe::caffe_cpu_axpby(64, (double)1.f, weights_dense2_b, (double)1.f, y2);
        
        //relu
        for (int i = 0; i < 64; ++i)
            y2[i] = std::max(y2[i], double(0));

        double* y3 = new double[64]{0};
        caffe_cpu_gemv(CblasNoTrans, 6, 64, (double)1.f, weights_final_w, y2, (double)0., y3);
        caffe::caffe_cpu_axpby(6, (double)1.f, weights_final_b, (double)1.f, y3);
        
//         bib::Logger::PRINT_ELEMENTS(y3, 64, "ici ");
        
        delete y;
        delete y2;
        
        return y3;
    }
    
private:
    void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha, 
                                const double* A, const double* x, const double beta, double* y) {
        cblas_dgemv(CblasColMajor, TransA, M, N, alpha, A, M, x, 1, beta, y, 1);
    }
};

}

#endif
