#ifndef COMMONAE_H
#define COMMONAE_H

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

class CommonAE {
 public:
  std::ostream &display(std::ostream &out, bool display, bool dump) const {
    if (display) {
      _display(out);
    }

    if (dump) {
      _dump(out);
    }

    return out;
  }

  virtual void unique_invoke(boost::property_tree::ptree *,
                             boost::program_options::variables_map *) {}

 protected:
  virtual void _display(std::ostream &) const {}

  virtual void _dump(std::ostream &) const {}
};

#endif  // COMMONAE_H
