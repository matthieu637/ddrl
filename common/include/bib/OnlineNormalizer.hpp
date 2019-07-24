#ifndef ONLINENORMALIZER_HPP
#define ONLINENORMALIZER_HPP

#include <vector>

namespace bib {

class OnlineNormalizer{
public:
  OnlineNormalizer(uint _size) : data_number(0), mean(_size, 0), var(_size, 0), subvar(_size, 0){}
  
  void transform(std::vector<double>& output, const std::vector<double>& x){
    update_mean_var(x);
  
    for(uint i=0; i < x.size(); i++){
        output[i] = (x[i] - mean[i]);

        if(std::abs(var[i]) >= 1e-6)
          output[i] /= sqrt(var[i]);
    }
  }
  
  void transform_with_clip(std::vector<double>& output, const std::vector<double>& x, double clip=5) {
    update_mean_var(x);
  
    for(uint i=0; i < x.size(); i++){
        output[i] = (x[i] - mean[i]);

        if(std::abs(var[i]) >= 1e-6)
          output[i] /= sqrt(var[i]);
        
        if(output[i] > clip)
          output[i]=clip;
        else if (output[i] < -clip)
          output[i]=-clip;
    }
  }
  
  void update_mean_var(const std::vector<double>& x){
    double dndata_number = data_number + 1;
    double ddata_number = data_number;
    for(uint i=0; i < x.size(); i++){
      mean[i] = (mean[i] * ddata_number + x[i])/dndata_number;
    
      subvar[i] = (subvar[i] * ddata_number + (x[i]*x[i]))/dndata_number;
      var[i] = subvar[i] - (mean[i]*mean[i]);
    }
    
    data_number ++;
  }
  
  ulong data_number;
  std::vector<double> mean;
  std::vector<double> var;
  std::vector<double> subvar;
};

}
#endif
