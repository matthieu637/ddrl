#ifndef KERNEL_HPP_INCLUDED
#define KERNEL_HPP_INCLUDED

#include <EigenConfig.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>

class Kernel {
public :
    Kernel(int n_kernels_per_dims,int n_dims);
    Kernel();
    void setWeights(Eigen::VectorXf weights);
    float getValue(const std::vector<float>& state);
  const float PI = 3.14159265358979f;
private :
    int m_nKernelsPerDim;
    int m_nDims;
    int m_nKernels;
    Eigen::MatrixXf m_centers;
    Eigen::VectorXf m_widths;
    Eigen::VectorXf m_weights;

};

#endif // KERNEL_HPP_INCLUDED
