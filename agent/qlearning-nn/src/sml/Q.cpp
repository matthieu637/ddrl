#include "sml/Q.hpp"

#include <fstream>
#include <vector>
#include "boost/serialization/vector.hpp"

#include "bib/Utils.hpp"
#include "bib/XMLEngine.hpp"
#include "bib/Seed.hpp"
#include "bib/Logger.hpp"

namespace sml {

QTable::QTable(const StateTemplate *stmp, const ActionTemplate *atmp)
  : stmpl(stmp), atmpl(atmp) {
  shouldDeleteStmpl = false;
  map = new hashmap(stmpl->sizeNeeded() * atmpl->sizeNeeded());
  //     LOG_DEBUG("sized " << stmpl->sizeNeeded() * atmpl->sizeNeeded() <<  " "
  //     << map->size() );

  for (unsigned int i = 0; i < stmpl->sizeNeeded(); i++) {
    for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++) {
      map->at(i * atmpl->sizeNeeded() + j) = 0.L;
    }
  }
}
QTable::QTable(const ActionTemplate *atmp) : atmpl(atmp) {
  shouldDeleteStmpl = true;
  stmpl = new StateTemplate({""}, {1});  // for file saving
  map = new hashmap(atmpl->sizeNeeded());

  for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++) map->at(j) = 0.L;
}

QTable::QTable(const QTable &q) : stmpl(q.stmpl), atmpl(q.atmpl) {
  shouldDeleteStmpl = false;
  map = new hashmap(*q.map);
}

QTable::~QTable() {
  if (shouldDeleteStmpl) delete stmpl;
  delete map;
}

DAction *QTable::argmax(const DState &name) const {
  unsigned int hashState = name.hash();
  unsigned int beginRange = hashState * atmpl->sizeNeeded();

  unsigned int imax = bib::Seed::unifRandInt(atmpl->sizeNeeded()-1);
  //      LOG_DEBUG(imax << " " << hashState << " " << atmpl->sizeNeeded() << "
  //      " << beginRange << " " << name << " " << map->size());
  for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++)
    if (map->at(beginRange + imax) < map->at(beginRange + j)) imax = j;

  return new DAction(atmpl, imax);
}
DAction *QTable::argmax() const {
  unsigned int imax = bib::Seed::unifRandInt( atmpl->sizeNeeded()-1);
  // unsigned int imax = 0;
  //     LOG_DEBUG(imax << " " << hashState << " " << atmpl->sizeNeeded() << " "
  //     << name["angle"] << " " << name["distance"] );
  for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++)
    if (map->at(imax) < map->at(j)) imax = j;

  return new DAction(atmpl, imax);
}
double QTable::max() const {
  unsigned int imax = bib::Seed::unifRandInt( atmpl->sizeNeeded()-1);
  for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++)
    if (map->at(imax) < map->at(j)) imax = j;

  return map->at(imax);
}
DAction *QTable::argmax(const std::vector<int> *action_time,
                        double gamma) const {
  unsigned int imax = bib::Seed::unifRandInt( atmpl->sizeNeeded()-1);
  double _max = map->at(imax) * powf(gamma, action_time->at(imax));

  for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++) {
    double nmax = map->at(j) * powf(gamma, action_time->at(j));
    if (_max < nmax) {
      imax = j;
      _max = nmax;
    }
  }
  return new DAction(atmpl, imax);
}

double QTable::min() const {
  unsigned int imin = bib::Seed::unifRandInt( atmpl->sizeNeeded()-1);
  for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++)
    if (map->at(imin) > map->at(j)) imin = j;

  return map->at(imin);
}
double QTable::operator()(const DState &s, const DAction &a) const {
  return this->operator()(s.hash(), a.hash());
}
double &QTable::operator()(const DState &s, const DAction &a) {
  return this->operator()(s.hash(), a.hash());
}

double QTable::operator()(const DState *s, const DAction *a) const {
  return this->operator()(*s, *a);
}
double &QTable::operator()(const DState *s, const DAction *a) {
  return this->operator()(*s, *a);
}

double QTable::operator()(const DAction &a) const {
  return this->operator()(0, a.hash());
}
double &QTable::operator()(const DAction &a) {
  return this->operator()(0, a.hash());
}

double QTable::operator()(unsigned int s, unsigned int a) const {
  unsigned int beginRange = s * atmpl->sizeNeeded();

  //     LOG_DEBUG("acces at " << beginRange + a);
  return (*map)[beginRange + a];
}
double &QTable::operator()(unsigned int s, unsigned int a) {
  unsigned int beginRange = s * atmpl->sizeNeeded();

  //     LOG_DEBUG("set at " << beginRange + a);
  return (*map)[beginRange + a];
}

void QTable::save(boost::archive::xml_oarchive *xml) {
  *xml << make_nvp("QTable", map);
}

void QTable::load(boost::archive::xml_iarchive *xml) {
  hashmap *obj = new hashmap;
  *xml >> make_nvp("QTable", *obj);
  map = obj;
}

void QTable::print(bool perState) const {
  //     std::cout.set(ios::fixed, ios::doublefield);
  std::cout.precision(2);

  if (perState) {
    for (unsigned int i = 0; i < stmpl->sizeNeeded(); i++) {
      for (unsigned int j = 0; j < atmpl->sizeNeeded(); j++)
        std::cout << map->at(i * atmpl->sizeNeeded() + j) << " ";
      std::cout << std::endl;
    }
  }
  {
    for (unsigned int i = 0; i < atmpl->sizeNeeded(); i++) {
      for (unsigned int j = 0; j < stmpl->sizeNeeded(); j++)
        std::cout << map->at(i + atmpl->sizeNeeded() * j) << " ";
      std::cout << std::endl;
    }
  }
  std::cout << "####################################################"
            << std::endl;
}
}  // namespace sml
