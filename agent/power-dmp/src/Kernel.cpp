#include "Kernel.hpp"

using namespace Eigen;

Kernel::Kernel() : m_nKernelsPerDim(4), m_nDims(2){}
Kernel::Kernel(int nKernelsPerDim, int nDims) : m_weights(256), m_widths(nDims*2), m_nKernelsPerDim(nKernelsPerDim), m_nDims(nDims)
{
    m_nKernels = std::pow(m_nKernelsPerDim, m_nDims*2);

    m_centers.resize(m_nDims*2,m_nKernels);
    Eigen::MatrixXf centersLinSpaced(m_nKernelsPerDim,m_nDims*2);
    m_weights = Eigen::VectorXf::Zero(m_nKernels);
    m_widths(0) = PI/(m_nKernelsPerDim+1);
    m_widths(1) = 20*PI/(m_nKernelsPerDim+1);
    m_widths(2) = 2*PI/(m_nKernelsPerDim+1);
    m_widths(3) = 40*PI/(m_nKernelsPerDim+1);
    m_widths *= 0.65;
    m_widths = m_widths.array().square().inverse();
    centersLinSpaced.col(0) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,0+m_widths(0),PI-m_widths(0)).transpose();
    centersLinSpaced.col(1) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,-10*PI+m_widths(1),10*PI+m_widths(1)).transpose();
    centersLinSpaced.col(2) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,-PI+m_widths(2),PI-m_widths(2)).transpose();
    centersLinSpaced.col(3) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,-20*PI+m_widths(3),20*PI-m_widths(3)).transpose();

    unsigned int r;
    for (unsigned int i = 0; i < m_nKernels; i++){
        r=i;
        for(unsigned int dim=0; dim<m_nDims; dim++){
            div_t q = div(r,m_nKernelsPerDim);
            m_centers(dim, i) =  centersLinSpaced(q.rem, dim);
            r=q.quot;
        }
    }

}
float Kernel::getValue(const std::vector<float>& sensors){
        //std::cout << "hiii" << std::endl;
        //std::cout << m_centers << std::endl;
    VectorXf psi(m_nKernels);
    VectorXf state(m_nDims*2);
    bool inv = false;
    //Map<Vector4f> state(sensors);
    state(0)=sensors[0];
    state(1)=sensors[1];
    state(2)=sensors[2];
    state(3)=sensors[3];
    if(state(0)<0){
        state*=-1;
        inv=true;
    }
for(unsigned int k=0; k<m_nKernels;k++){

    float psi_tmp = (state.array()-m_centers.col(k).array()).square().matrix().dot(m_widths);
    psi(k) = exp(-0.5*psi_tmp);
}
float retour;
//if(state(3)>5.0f)
//retour = (m_weights.array()*psi.array()*(state(3)+0.0001f)).sum()/(psi.sum()+0.00000000001f);
//else
retour = (m_weights.array()*psi.array()).sum()/(psi.sum()+0.00000000001f);
//float retour = (m_weights.array()*psi.array()).sum()/(psi.sum()+0.00000000001f);
//cout << state(3) << state(0) << endl;

if(retour>1)
    retour=1;
else if(retour<-1)
    retour=-1;
    if(inv)
        retour*=-1;
return retour;
//return 1;
}
void Kernel::setWeights(VectorXf weights){
    m_weights =weights;
}

