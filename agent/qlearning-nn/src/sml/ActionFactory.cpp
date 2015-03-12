#include "sml/ActionFactory.hpp"

#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <vector>
#include "boost/serialization/vector.hpp"
#include "boost/serialization/utility.hpp"
#include "boost/filesystem.hpp"

#include "bib/Utils.hpp"
#include "bib/XMLEngine.hpp"

namespace sml {

const list_tlaction& ActionFactory::getActions() const {
  return actions;
}
int ActionFactory::getActionsNumber() const {
  return numberAction;
}
class malformedfile : public std::exception {
  virtual const char* what() const throw() {
    return "The action file is malformed !";
  }
} malformedfile_ex;

void ActionFactory::write(const list_tlaction& ac, const std::string& file) {
  bib::XMLEngine::save<list_tlaction>(ac, "actions", file);
}

void ActionFactory::read(const std::string& file) {
  list_tlaction* ptr = bib::XMLEngine::load<list_tlaction>("actions", file);
  actions.clear();
  for (auto it = ptr->cbegin(); it != ptr->cend(); ++it)
    actions.push_back(*it);
  delete ptr;
  numberAction = actions.size();
}

void ActionFactory::injectArgs(int _numberAction) {
  numberAction = _numberAction;
}

void ActionFactory::injectArgs(const std::string& key, int numberMotor,
                               unsigned int numberAction) {
  if (!boost::filesystem::exists(key)) {
    std::cout << "Can't find the action file! (" << key << ")" << std::endl;
    exit(1);
  }

  ifstream myfile(key);

  string line;

  TemporalLinearAction ac;

  try {
    int i = 1;
    while (std::getline(myfile, line)) {
      std::stringstream sstream(line);
      std::string value;

      TemporalLinearMotor tlm;

      if (std::getline(sstream, value, ' '))
        tlm.a = stof(value);
      else
        throw malformedfile_ex;

      if (std::getline(sstream, value, ' '))
        tlm.b = stof(value);
      else
        throw malformedfile_ex;

      if (std::getline(sstream, value, ' '))
        ac.temporal_extension = stoi(value);
      else
        throw malformedfile_ex;

      ac.motors.push_back(tlm);

      if (i % numberMotor == 0 && i != 1) {
        actions.push_back(ac);
        if (actions.size() >= numberAction) break;

        ac.motors.clear();
        ac.temporal_extension = -1;
      }

      i++;
    }
  } catch (const std::exception& e) {
    LOG_ERROR(e.what());
    exit(1);
  }

  myfile.close();

  injectArgs(actions.size());
}

void ActionFactory::gridAction(unsigned int numberMotor, unsigned int actionPerMotor) {
  ASSERT(actionPerMotor >= 3, "invalid number of action per motor : {-1 0 1}");
  ASSERT(numberMotor <= 2, "not implemented for more than 2 motors");

#ifndef NDEBUG
  if (actionPerMotor % 2 == 0)
    LOG_INFO("Number of action per motor is a pair number so there is no neutral action");
#endif

  actions.clear();

  std::vector<TemporalLinearMotor> tlm(actionPerMotor);
  float factor = 2.f / (static_cast<float>(actionPerMotor) - 1.f);

  for (unsigned int i = 0; i < actionPerMotor; i++) {
    tlm[i].a = 0.f;
    tlm[i].b = (factor * (static_cast<float>(i))) - 1.f;
  }

  numberAction = static_cast<unsigned int>(pow(actionPerMotor, numberMotor));
  std::vector<TemporalLinearAction> ac(numberAction);

  for (unsigned int i = 0; i < numberAction; i++) {
    ac[i].temporal_extension = 1;
    ac[i].motors.push_back(tlm[i % actionPerMotor]);
  }

  for (unsigned int m = 1; m < numberMotor; m++) {
    for (unsigned int i = 0; i < numberAction; i++) {
      ac[i].motors.push_back(tlm[ ((i / actionPerMotor) + i) % actionPerMotor]);
    }
  }

  for (auto it = ac.cbegin(); it != ac.cend(); ++it) {
//       LOG_DEBUG(std::left << std::setw(5) << std::fixed << std::setprecision(1) <<
//       it->motors[0].b << std::left << std::setw(5) << std::fixed << std::setprecision(1) << it->motors[1].b <<
//       std::left << std::setw(5) << std::fixed << std::setprecision(1) << it->motors[2].b);

    actions.push_back(*it);
  }

  numberAction = actions.size();
}

void ActionFactory::randomLinearAction(int numberMotor, int timestepMin,
                                       int timestepMax,
                                       unsigned int numberAction) {
  actions.clear();

  assert(numberAction > 0);
  //     LOG_DEBUG("redefining actions");

  do {
    TemporalLinearAction ac;
    ac.temporal_extension = bib::Utils::randin(timestepMin, timestepMax);
    for (int motor = 0; motor < numberMotor; motor++) {
      float y1 = bib::Utils::rand01();
      float y2 = bib::Utils::rand01();

      TemporalLinearMotor tlm;
      tlm.a = static_cast<float>((y2 - y1) / ac.temporal_extension);
      tlm.b = y2;
      ac.motors.push_back(tlm);
    }

    actions.push_back(ac);
    ac.motors.clear();
  } while (actions.size() < numberAction);
}

void ActionFactory::randomFixedAction(int numberMotor,
                                      unsigned int numberAction,
                                      int timestepMin, int timestepMax) {
  actions.clear();

  do {
    TemporalLinearAction ac;
    for (int motor = 0; motor < numberMotor; motor++) {
      TemporalLinearMotor tlm;
      tlm.a = 0;
      tlm.b = bib::Utils::rand01();
      ac.motors.push_back(tlm);
    }
    ac.temporal_extension = bib::Utils::randin(timestepMin, timestepMax);
    actions.push_back(ac);
    ac.motors.clear();
  } while (actions.size() < numberAction);
}

void ActionFactory::randomLinearAction(int numberMotor, int timestepMin,
                                       int timestepMax) {
  randomLinearAction(numberMotor, timestepMin, timestepMax, numberAction);
}

void ActionFactory::randomFixedAction(int numberMotor, int timestepMin,
                                      int timestepMax) {
  randomFixedAction(numberMotor, numberAction, timestepMin, timestepMax);
}

std::vector<float>* ActionFactory::computeOutputs(
  const DAction* ac, int timestep, const sml::list_tlaction& actions) {
  int ac_id = ac->get(0);
  const TemporalLinearAction& action = actions.at(ac_id);

  int nb_motors = action.motors.size();
  std::vector<float>* outputs = new std::vector<float>(nb_motors, 0.5);

  for (int motor = 0; motor < nb_motors; motor++)
    outputs->operator[](motor) =
      action.motors[motor].a * timestep + action.motors[motor].b;

  return outputs;
}
}  // namespace sml
