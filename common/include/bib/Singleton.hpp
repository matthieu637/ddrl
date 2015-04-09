#ifndef SINGLETON_HPP
#define SINGLETON_HPP

#include <list>
#include <memory>

#include "Assert.hpp"

///
// /\file Singleton.hpp
// /\brief Template de base pour le pattern singleton
///
///
/// \example Pour l'utiliser il suffit de faire hériter la classe par celle-ci
///          et d'ajouter : friend class bib::Singleton<MaClasse>;
///
///

namespace bib
{

template <class T>
class Singleton
{
public:
    ///
    // /\brief Méthode statique et public pour récupérer l'instance du singleton
    ///
    static T* getInstance() {
        ASSERT(_singleton != nullptr, "singleton never created");
        return _singleton.get();
    }

    ///
    // /\brief Constructeur privée/protected pour empécher l'instanciation
    /// n'importe où
    ///
protected:
    Singleton() {}
public:
    virtual ~Singleton() {}

private:
    static std::shared_ptr<T> _singleton;
};

template <class T>
std::shared_ptr<T> Singleton<T>::_singleton = std::shared_ptr<T>(new T);
}  // namespace bib

#endif
