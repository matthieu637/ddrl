#ifndef INIPARSER_H
#define INIPARSER_H

#include <iostream>
#include <string>
#include <vector>
#include "boost/lexical_cast.hpp"

namespace bib {

template <typename T>
std::vector<T>* to_array(const std::string& s) {
  std::vector<T>* result = new std::vector<T>;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    result->push_back(boost::lexical_cast<T>(item));
  return result;
}
}  // namespace bib
#endif  // INIPARSER_H
