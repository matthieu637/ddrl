
#ifndef Q_HPP
#define Q_HPP

///
///\file Q.hpp
///\brief un tableau à une dimension qui simule le fonctionnement d'un tableau
///"Etat X Action"
//

#include <string>
#include <vector>
#include "bib/XMLEngine.hpp"
#include "sml/Action.hpp"

using std::string;
using std::vector;

namespace sml {

typedef vector<double> hashmap;

class QTable {
 public:
  ///
  ///\brief Constructeur pour créer le tableau "Etat X Action"
  ///\param stmp : le modèle d'état
  ///     atmp : le modèle d'action
  QTable(const StateTemplate *stmp, const ActionTemplate *atmp);

  QTable(const QTable &q);

  ///
  ///\brief Constructeur pour créer le tableau "Action"
  ///\param atmp : le modèle d'action
  QTable(const ActionTemplate *atmp);

  ~QTable();

  ///
  ///\brief Retourner toutes les actions posiibles pour un état donné
  ///\param name : le nom d'état
  const hashmap *operator[](const DState &name) const;

  ///
  ///\brief Retourner l'actions maximale dans le tableau pour un état donné
  ///\param name : le nom d'état
  DAction *argmax(const DState &name) const;

  ///
  ///\brief Retourner l'actions maximale dans le tableau
  DAction *argmax() const;
  float max() const;
  DAction *argmax(const std::vector<int> *action_time, float gamma) const;

  DAction *boltzmann(float temperature) const;

  DAction *argmin() const;
  float min() const;

  ///
  ///\brief Renvoyer la valeur de (s,a) dans le tableau Q
  ///\param s : l'état
  ///     a : l'action
  double operator()(const DState &s, const DAction &a) const;
  double &operator()(const DState &s, const DAction &a);
  double operator()(const DState *s, const DAction *a) const;
  double &operator()(const DState *s, const DAction *a);

  ///
  ///\brief Renvoyer la valeur d'action "a" dans le tableau Q
  ///\param a : l'action
  double operator()(const DAction &a) const;
  double &operator()(const DAction &a);

  ///
  ///\brief Renvoyer la valeur dans le tableau Q avec des indices
  ///\param s : l'indice pour l'état
  ///     a : l'indice pour l'action
  double operator()(unsigned int s, unsigned int a) const;
  double &operator()(unsigned int s, unsigned int a);

  ///
  ///\brief Sauvegarder ce que l'algorithme a appris
  ///\param xml : le fichier XML
  void save(boost::archive::xml_oarchive *xml);

  ///
  ///\brief Charger ce que l'algorithme a appris
  ///\param xml : le fichier XML
  void load(boost::archive::xml_iarchive *xml);

  ///
  ///\brief Renvoyer toutes les actions
  hashmap *getWholeCouple();

  void print(bool perState = true) const;

 private:
  bool shouldDeleteStmpl;
  hashmap *map = nullptr;
  // TODO: could be slightly improve argmax performance by using multiset and
  // hashmap together

  const StateTemplate *stmpl;
  const ActionTemplate *atmpl;
};
} // namespace sml
#endif
