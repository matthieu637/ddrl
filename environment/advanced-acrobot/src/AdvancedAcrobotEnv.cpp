#include "AdvancedAcrobotEnv.hpp"

#include <string>
#include "boost/algorithm/string/case_conv.hpp"

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

ProblemDefinition* str2prob(const std::string& s1) {
  std::string s(s1);
  boost::algorithm::to_upper<>(s);
  if (s == "KEEPHIGH")
    return new KeepHigh;
  else if (s == "REACHLIMITPOORINFORMED")
    return new ReachLimitPoorInformed;
  else if (s == "REACHLIMITWELLINFORMED")
    return new ReachLimitWellInformed;
  else if (s == "REACHLIMITPOORINFORMEDMAX")
    return new ReachLimitPoorInformedMax;

  return new KeepHigh;
}
