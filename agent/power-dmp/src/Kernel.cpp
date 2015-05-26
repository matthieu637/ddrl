#include "Kernel.hpp"

using namespace Eigen;

Kernel::Kernel() : m_nKernelsPerDim(4), m_nDims(4){}
Kernel::Kernel(int nKernelsPerDim, int nDims) : m_widths(nDims), m_nDims(nDims)
{
    m_nKernelsPerDim.resize(nDims);

    //m_nKernelsPerDim << 2,2,2,2,2,2;
    //m_nKernelsPerDim << 3,3,3,3,3,3;
    m_nKernels=1;
    for(int i=0;i<nDims;i++){
        m_nKernelsPerDim[i]=nKernelsPerDim;
        m_nKernels*=m_nKernelsPerDim[i];
    }
    //m_nKernels = std::pow(m_nKernelsPerDim, m_nDims);

    m_centers.resize(m_nDims,m_nKernels);
    //Eigen::MatrixXf centersLinSpaced(m_nKernelsPerDim,m_nDims);
    //m_weights.resize(m_nKernels);
    m_weights = Eigen::VectorXf::Zero(m_nKernels);
/*
    m_widths(0) = PI/(m_nKernelsPerDim+1);
    m_widths(1) = 20*PI/(m_nKernelsPerDim+1);
    m_widths(2) = 2*PI/(m_nKernelsPerDim+1);
    m_widths(3) = 40*PI/(m_nKernelsPerDim+1);
    m_widths *= 0.60f;
    m_widths = m_widths.array().square().inverse();
    centersLinSpaced.col(0) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,0+m_widths(0),PI-m_widths(0)).transpose();
    centersLinSpaced.col(1) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,-10*PI+m_widths(1),10*PI+m_widths(1)).transpose();
    centersLinSpaced.col(2) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,-PI+m_widths(2),PI-m_widths(2)).transpose();
    centersLinSpaced.col(3) = VectorXf::LinSpaced(Sequential,m_nKernelsPerDim,-20*PI+m_widths(3),20*PI-m_widths(3)).transpose();
*/

    std::vector <VectorXf> centersSpaced;
    for(int i=0;i<nDims;i++){
        //if(i==0||i==2){
        if(i==0||i==2){
            //m_widths(i) = 1.0f/(m_nKernelsPerDim[i]-1.f);
            m_widths(i) = 1.0f/(m_nKernelsPerDim[i]-1);
        } else if(i==1){
            m_widths(i) = 2.0f/(m_nKernelsPerDim[i]-1);
        } else if(i%2==0){
            m_widths(i) = 2.0f/(m_nKernelsPerDim[i]-1);
        } else {
            m_widths(i) = 2.0f/(m_nKernelsPerDim[i]+1);
        }
        if(i==0||i==2)
            centersSpaced.push_back(VectorXf::LinSpaced(Sequential,m_nKernelsPerDim[i],0.f,1.f).transpose());
        else
            centersSpaced.push_back(VectorXf::LinSpaced(Sequential,m_nKernelsPerDim[i],-1.f,1.f).transpose());
    }

    m_widths *= 1.f;

    m_widths = m_widths.array().square().inverse();

    unsigned int r;
    for (unsigned int i = 0; i < m_nKernels; i++){
        r=i;
        std::cout << "Kernel " << i << " ";
        for(unsigned int dim=0; dim<m_nDims; dim++){
            div_t q = div(r,m_nKernelsPerDim[dim]);
            std::cout << "dim " << dim << "i " << q.rem << " ";
            //m_centers(dim, i) =  centersLinSpaced(q.rem, dim);
            m_centers(dim, i) =  centersSpaced[dim](q.rem);
            r=q.quot;
        }
        std::cout << std::endl;

    }

}

float Kernel::getValue(const std::vector<float>& sensors, const int& dim){

    VectorXf psi(m_nKernels);
    VectorXf state(dim);
    bool inv = false;

    for(int i=0;i<dim;i++){
        state(i)=sensors[i];
        if(i%2==0){
            while(state(i)>1.f) state(i)-=2.f;
            while(state(i)<-1.f) state(i)+=2.f;
        }
    }

    if(state(0)<0){
        state*=-1.0f;
        inv=true;
    }

for(unsigned int k=0; k<m_nKernels;k++){

    float psi_tmp = (state.array()-m_centers.col(k).array()).square().matrix().dot(m_widths);
    psi(k) = exp(-0.5f*psi_tmp);
}
float retour;

retour = (m_weights.array()*psi.array()).sum()/(psi.sum()+0.00000000001f);

if(retour>1)
    retour=1;
else if(retour<-1)
    retour=-1;
    if(inv){
        retour*=-1.f;
    }
return retour;

}
void Kernel::setWeights(const VectorXf weights){
    m_weights =weights;
}
float Kernel::getWeight(const int& index) const{
    return m_weights[index];
}
//unsigned int Kernel::getSize(){return m_nKernels}

