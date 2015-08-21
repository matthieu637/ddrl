#ifndef ACTIONFACTORY_H
#define ACTIONFACTORY_H

#include <string>
#include <vector>
#include "boost/serialization/nvp.hpp"
#include "boost/serialization/vector.hpp"
#include "bib/Singleton.hpp"
#include "Action.hpp"

namespace sml {

struct TemporalLinearMotor {
  double a;
  double b;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int) {
    ar &BOOST_SERIALIZATION_NVP(a);
    ar &BOOST_SERIALIZATION_NVP(b);
  }
};

struct TemporalLinearAction {
  std::vector<TemporalLinearMotor> motors;
  int temporal_extension;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int) {
    ar &BOOST_SERIALIZATION_NVP(temporal_extension);
    ar &BOOST_SERIALIZATION_NVP(motors);
  }
};

typedef std::vector<TemporalLinearAction> list_tlaction;

class ActionFactory : public bib::Singleton<ActionFactory> {
  friend class bib::Singleton<ActionFactory>;

 protected:
  ActionFactory() : actions(), numberAction(-1) {}

 public:
  const list_tlaction &getActions() const;
  int getActionsNumber() const;
  void injectArgs(const std::string &key, int numberMotor,
                  unsigned int numberAction);
  void injectArgs(int numberAction);

  void gridAction(unsigned int numberMotor, unsigned int actionPerMotor);

  void randomLinearAction(int numberMotor, int timestepMin, int timestepMax,
                          unsigned int _numberAction);

  void randomFixedAction(int numberMotor, unsigned int _numberAction,
                         int timestepMin, int timestepMax);

  void randomLinearAction(int numberMotor, int timestepMin, int timestepMax);

  void randomFixedAction(int numberMotor, int timestepMin, int timestepMax);

  void write(const list_tlaction &, const std::string &);

  void read(const std::string &);

  static std::vector<double> *computeOutputs(const DAction *ac, int timestep,
      const sml::list_tlaction &actions);

 private:
  list_tlaction actions;
  unsigned int numberAction;
};
}  // namespace sml
#endif  // ACTIONFACTORY_H
