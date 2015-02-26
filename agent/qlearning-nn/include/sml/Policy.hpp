#ifndef POLICY_HPP
#define POLICY_HPP

#include <string>
#include "boost/filesystem.hpp"
#include "boost/interprocess/sync/scoped_lock.hpp"
#include "boost/interprocess/sync/named_mutex.hpp"
#include "bib/XMLEngine.hpp"
#include "bib/Logger.hpp"
#include "Action.hpp"


using std::string;
using namespace boost::interprocess;

namespace sml {

struct RLParam {
  float epsilon;
  float alpha;
  float gamma;
  unsigned int repeat_replay;
  int memory_size;

  int hidden_unit;
  float activation_stepness;
  std::string activation;
};

struct LearnReturn {
  DAction *ac;
  bool gotGreedy;
};

template <class State>
class Policy {
 public:
  Policy(RLParam param) : param(param) {}
  virtual ~Policy() {}

  virtual DAction *decision(const State &st, bool greedy) = 0;
  virtual void startEpisode(const State &s, const DAction &a) = 0;

  virtual LearnReturn _learn(const State &s, double reward, bool goal) = 0;

  DAction *learn(const State &s, double reward, bool goal) {
    return _learn(s, reward, goal).ac;
  }

  virtual Policy<State> *copyPolicy() = 0;

  RLParam &getParams() {
    return param;
  }

  ///
  ///\brief Sauvegarder les données de l'algorithme
  ///\param chemin : l'adresse où on sauvegarde
  virtual void write(const string &chemin) {
    //         named_mutex mutex( open_or_create, chemin.c_str());

    LOG_INFO("Enregistrement du fichier XML " << chemin);
    //         mutex.lock();

    ofstream outputFile(chemin);
    assert(outputFile.good());
    xml_oarchive xml(outputFile);
    save(&xml);
    outputFile.close();
    //         mutex.unlock();
  }

  ///
  ///\brief Charger les données de l'algorithme
  ///\param chemin : l'adresse où on charge
  virtual void read(const string &chemin) {
    if (!boost::filesystem::exists(chemin)) {
      LOG_DEBUG(chemin << " n'existe pas.");
    } else {
      //             named_mutex mutex( open_or_create, chemin.c_str());
      //             mutex.lock();

      ifstream inputFile(chemin);
      assert(inputFile.good());
      xml_iarchive xml(inputFile);
      load(&xml);
      inputFile.close();
      //             mutex.unlock();
    }
  }

 protected:
  ///
  ///\brief Sauvegarder ce que l'algorithme a appris
  ///\param xml : le fichier XML
  virtual void save(boost::archive::xml_oarchive *xml) = 0;

  ///
  ///\brief Charger ce que l'algorithme a appris
  ///\param xml : le fichier XML
  virtual void load(boost::archive::xml_iarchive *xml) = 0;

 protected:
  RLParam param;
};

typedef Policy<DState> DPolicy;
}

#endif  // MCARTASK_H
