#include "AdvancedAcrobotEnv.hpp"

#include <boost/algorithm/string/case_conv.hpp>

std::istream& operator>>(std::istream& istream, bone_joint& v) {
  std::string s;
  istream >> s;
  boost::algorithm::to_upper<>(s);
  if (s == "HINGE")
    v = HINGE;
  else
    v = SLIDER;
  return istream;
}
