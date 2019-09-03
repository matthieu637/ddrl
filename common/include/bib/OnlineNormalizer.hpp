#ifndef ONLINENORMALIZER_HPP
#define ONLINENORMALIZER_HPP

#include <vector>
#include "bib/XMLEngine.hpp"

namespace bib {

class OnlineNormalizer{
public:
  OnlineNormalizer(uint _size) : data_number(_size, 0), mean(_size, 0), var(_size, 0), subvar(_size, 0){}
  
  void transform(std::vector<double>& output, const std::vector<double>& x, bool learn) {
    if(learn)
      update_mean_var(x);
  
    for(uint i=0; i < x.size(); i++){
        output[i] = (x[i] - mean[i]);

        if(std::abs(var[i]) >= 1e-6)
          output[i] /= sqrt(var[i]);
    }
  }
  
  void transform_with_clip(std::vector<double>& output, const std::vector<double>& x, bool learn, double clip=5) {
    if(learn)
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
  
    void transform_with_double_clip(std::vector<double>& output, const std::vector<double>& x, bool learn, double clip1=200, double clip2=5) {
    for(uint i=0; i < x.size(); i++) {
      output[i] = x[i];
      
      if(output[i] > clip1)
        output[i]=clip1;
      else if (output[i] < -clip1)
        output[i]=-clip1;
    }
    if(learn)
      update_mean_var(output);
  
    for(uint i=0; i < x.size(); i++){
        output[i] = (x[i] - mean[i]);

        if(std::abs(var[i]) >= 1e-6)
          output[i] /= sqrt(var[i]);
        
        if(output[i] > clip2)
          output[i]=clip2;
        else if (output[i] < -clip2)
          output[i]=-clip2;
    }
  }
  
  void update_batch_clip_before(const std::vector<double>& x, uint start_col=0, double clip1=200) {
    std::vector<double> output(x.size());
    
    for(uint i=0; i < x.size(); i++) {
      output[i] = x[i];
      
      if(output[i] > clip1)
        output[i]=clip1;
      else if (output[i] < -clip1)
        output[i]=-clip1;
    }
    
    update_mean_var(output, start_col);
  }
  
  void update_mean_var(const std::vector<double>& x, uint start_col=0) {
    for(uint i=start_col; i < x.size(); i++) {
      double dndata_number = data_number[i] + 1;
      double ddata_number = data_number[i];
      
      mean[i] = (mean[i] * ddata_number + x[i])/dndata_number;
    
      subvar[i] = (subvar[i] * ddata_number + (x[i]*x[i]))/dndata_number;
      var[i] = subvar[i] - (mean[i]*mean[i]);
      data_number[i] ++;
    }
  }
  
  void copyFrom(const OnlineNormalizer& on) {
    ASSERT(on.data_number.size() == data_number.size(), "size pb");
    
     std::copy(on.data_number.begin(), on.data_number.end(), data_number.begin());
     std::copy(on.mean.begin(), on.mean.end(), mean.begin());
     std::copy(on.var.begin(), on.var.end(), var.begin());
     std::copy(on.subvar.begin(), on.subvar.end(), subvar.begin());
  }
  
private:
  friend class bib::XMLEngine;
  OnlineNormalizer(){}
  
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(data_number);
    ar& BOOST_SERIALIZATION_NVP(mean);
    ar& BOOST_SERIALIZATION_NVP(var);
    ar& BOOST_SERIALIZATION_NVP(subvar);
  }
  
  std::vector<ulong> data_number;
  std::vector<double> mean;
  std::vector<double> var;
  std::vector<double> subvar;
};

}
#endif
