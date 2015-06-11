#include "Kernel.hpp"

using namespace Eigen;

Kernel::Kernel() : n_basis_per_dim(4), n_dims(4){}
Kernel::Kernel(int _n_basis_per_dim, int _n_dims) : widths(_n_dims), n_dims(_n_dims)
{
    n_basis_per_dim.resize(_n_dims);

    //n_basis_per_dim << 2,2,2,2,2,2;
    //n_basis_per_dim << 3,3,3,3,3,3;
    n_basis=1;
    for(int i=0;i<n_dims;i++){
        n_basis_per_dim[i]=_n_basis_per_dim;
        n_basis*=n_basis_per_dim[i];
    }

    centers.resize(n_dims,n_basis);
    weights = Eigen::VectorXf::Zero(n_basis);
/*
    widths(0) = PI/(n_basis_per_dim+1);
    widths(1) = 20*PI/(n_basis_per_dim+1);
    widths(2) = 2*PI/(n_basis_per_dim+1);
    widths(3) = 40*PI/(n_basis_per_dim+1);
    widths *= 0.60f;
    widths = widths.array().square().inverse();
    centersLinSpaced.col(0) = VectorXf::LinSpaced(Sequential,n_basis_per_dim,0+widths(0),PI-widths(0)).transpose();
    centersLinSpaced.col(1) = VectorXf::LinSpaced(Sequential,n_basis_per_dim,-10*PI+widths(1),10*PI+widths(1)).transpose();
    centersLinSpaced.col(2) = VectorXf::LinSpaced(Sequential,n_basis_per_dim,-PI+widths(2),PI-widths(2)).transpose();
    centersLinSpaced.col(3) = VectorXf::LinSpaced(Sequential,n_basis_per_dim,-20*PI+widths(3),20*PI-widths(3)).transpose();
*/

    std::vector <VectorXf> centersSpaced;
    for(int i=0;i<_n_dims;i++){
        if(i==0||i==2){
            //widths(i) = 1.0f/(n_basis_per_dim[i]-1.f);
            widths(i) = 2.0f/(n_basis_per_dim[i]);
        } else if(i==1){
            widths(i) = 2.0f/(n_basis_per_dim[i]);
        } else if(i%2==0){
            widths(i) = 2.0f/(n_basis_per_dim[i]);
        } else {
            widths(i) = 2.0f/(n_basis_per_dim[i]);
        }
        if(i==0||i==2)
            centersSpaced.push_back(VectorXf::LinSpaced(Sequential,n_basis_per_dim[i],-0.5f,0.5f).transpose());
        else
            centersSpaced.push_back(VectorXf::LinSpaced(Sequential,n_basis_per_dim[i],-0.5f,0.5f).transpose());
    }

    widths *= 0.55f;

    widths = widths.array().square().inverse();

    unsigned int r;
    for (unsigned int i = 0; i < n_basis; i++){
        r=i;
        //std::cout << "Kernel " << i << " ";
        for(unsigned int dim=0; dim<n_dims; dim++){
            div_t q = div(r,n_basis_per_dim[dim]);
            //std::cout << "dim " << dim << "i " << q.rem << " ";
            centers(dim, i) =  centersSpaced[dim](q.rem);
            r=q.quot;
        }
        //std::cout << std::endl;
    }

}

float Kernel::getValue(const std::vector<float>& sensors, const int& dim){

    VectorXf psi(n_basis);
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
        //inv=true;
    }

for(unsigned int k=0; k<n_basis;k++){

    float psi_tmp = (state.array()-centers.col(k).array()).square().matrix().dot(widths);
    psi(k) = exp(-0.5f*psi_tmp);
}
float retour;

retour = (weights.array()*psi.array()).sum()/(psi.sum()+0.00000000001f);

if(retour>1)
    retour=1;
else if(retour<-1)
    retour=-1;
    if(inv){
        retour*=-1.f;
    }
return retour;

}
void Kernel::setWeights(const VectorXf _weights){
    weights =_weights;
}
float Kernel::getWeight(const int& index) const{
    return weights[index];
}
//unsigned int Kernel::getSize(){return n_basis}

