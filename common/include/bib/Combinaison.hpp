#ifndef COMBINAISON_HPP
#define COMBINAISON_HPP

#include <vector>

namespace bib {

class Combinaison {
 public:
  template<typename Function1>
  static void continuous(Function1 && eval, uint dimension, double min=-1.f, double max=1.f, uint discretization=30) {
    
    std::vector<double> x(dimension, min);
    _continuous<Function1>(x, 0, eval, dimension, min, max, discretization);
  }
  
  private:
  template<typename Function1>
  static void _continuous(std::vector<double>& x, uint current, Function1 && eval, uint dimension, double min, double max, uint discretization) {
    
    bool last_one = current == dimension - 1;
    double step = (max-min)/discretization;
    
    for(double a=min; a <= max; a += step){
      x[current] = a;
      if(last_one)
        eval(x);
      else
        _continuous<Function1>(x, current+1, eval, dimension, min, max, discretization);
    }
  }
 
};

}

#endif
