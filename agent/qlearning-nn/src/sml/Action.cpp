#include "sml/Action.hpp"

#include <string>
#include <list>

#include "bib/Logger.hpp"

namespace sml {

DAction::DAction(const ActionTemplate* temp, const std::list<int>& vals) {
  assert((int)vals.size() == temp->actionNumber());

  this->templ = temp;

  values = new int[templ->actionNumber()];
  int i = 0;
  for (std::list<int>::const_iterator it = vals.begin(); it != vals.end();
       ++it) {
    values[i] = *it;
    i++;
  }
  computehash();
}

DAction::DAction(const ActionTemplate* temp, int value) {
  this->templ = temp;

  hashmem = value;
  values = new int[templ->actionNumber()];

  list<int>::const_iterator it = templ->sizesActions()->begin();
  ++it;  // always ignore first multiplier
  for (int i = 0; i < templ->actionNumber(); i++) {
    int multiplier = 1;

    for (; it != templ->sizesActions()->end(); ++it)  // compute multiplier
      multiplier *= *it;

    //  LOG_DEBUG(multiplier << " " << value << " " << (int) ( value /
    //  multiplier ));

    values[i] = (int)(value / multiplier);
    value -= values[i] * multiplier;

    if (temp->sizeNeeded() == 3)
      //  LOG_DEBUG(values[i] << " " << value << " " << this->get("motor"));

      for (int j = 0; j < (templ->actionNumber() - 1) - (i + 1); j++)  // refill
        --it;
  }
}

DAction::DAction(const DAction& a) {
  this->templ = a.templ;
  hashmem = a.hashmem;
  int size = templ->actionNumber();
  values = new int[size];
  for (int i = 0; i < size; i++) values[i] = a.values[i];
}

DAction::~DAction() {
  delete[] values;
}

int DAction::get(const string& name) const {
  return values[templ->indexFor(name)];
}
int DAction::get(int index) const {
  return values[index];
}
int DAction::operator[](const string& name) const {
  return get(name);
}
void DAction::set(const string& name, int value) {
  values[templ->indexFor(name)] = value;
  computehash();
}

unsigned int DAction::hash() const {
  return hashmem;
}
void DAction::copyValuesOf(const DAction& ac) {
  //     LOG_DEBUG(*this << " copying " << ac);
  const boost::unordered_map<string, int>* names = ac.templ->getActionNames();
  for (ActionTemplate::nameIterator it = names->cbegin(); it != names->cend();
       ++it) {
    //  LOG_DEBUG(it->first << " " << ac[it->first]);
    this->set(it->first, ac[it->first]);
  }
  //     LOG_DEBUG(*this);
}

void DAction::computehash() {
  unsigned int hash = 0;
  list<int>::const_iterator it = templ->sizesActions()->begin();
  ++it;  // always ignore first multiplier

  for (int i = 0; i < templ->actionNumber(); i++) {
    int multiplier = 1;

    for (; it != templ->sizesActions()->end(); ++it)  // compute multiplier
      multiplier *= *it;

    hash += values[i] * multiplier;

    for (int j = 0; j < (templ->actionNumber() - 1) - (i + 1); j++)  // refill
      --it;
  }

  hashmem = hash;
}

bool DAction::operator==(const DAction& ac) const {
  if (*(ac.templ) == *(templ)) return hash() == ac.hash();

  return false;
}
bool DAction::operator<(const DAction& ac) const {
  return hash() < ac.hash();
}
void DAction::print(std::ostream& flux) const {
  flux << "{";
  for (int i = 0; i < templ->actionNumber(); i++) {
    flux << values[i];
    if (i + 1 < templ->actionNumber()) flux << ",";
  }
  flux << "}";
}

std::ostream& operator<<(std::ostream& stream, const sml::DAction& ac) {
  ac.print(stream);
  return stream;
}
} // namespace sml
