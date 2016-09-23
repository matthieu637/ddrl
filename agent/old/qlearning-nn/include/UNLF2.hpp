#ifndef UNLF2_HPP
#define UNLF2_HPP

#include <vector>

#include "opt++/NLF.h"

namespace OPTPP {

using NEWMAT::ColumnVector;

void init_never_call(int ndim, ColumnVector& x);

typedef void (*INITFCNU)(int, NEWMAT::ColumnVector&, const std::vector<double>&);

void init_called(int ndim, ColumnVector&, const std::vector<double>&);

class UNLF2 : public NLF2 {
 public:
  UNLF2(int ndim, USERFCN2V f, const std::vector<double>& _initial, CompoundConstraint* constraint = 0, void* v = 0):
    NLF2(ndim, f, init_never_call, constraint, v), initial(_initial) {
    init_fcnu = init_called;
  }

  void initFcn() override {
    if (init_flag == false) {
      init_fcnu(dim, mem_xc, initial);
      init_flag = true;
    } else {
      cerr << "NLF2:initFcn: Warning - initialization called twice\n";
      init_fcnu(dim, mem_xc, initial);
    }
  }

 protected:
  const std::vector<double>& initial;
  INITFCNU init_fcnu;
};

}

#endif
