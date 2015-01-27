#ifndef SINGLETON_HPP
#define SINGLETON_HPP

#include "Assert.hpp"

///
///\file Singleton.hpp
///\brief Template de base pour le pattern singleton
///
///
/// \example Pour l'utiliser il suffit de faire hériter la classe par celle-ci
///          et d'ajouter : friend class bib::Singleton<MaClasse>;
///
///

namespace bib {

template <class T>
class Singleton
{


public:
///
///\brief Méthode statique et public pour récupérer l'instance du singleton
///
    static T* getInstance()
    {

        ASSERT(_singleton != nullptr, "singleton never created");
        return _singleton;
    }
    
    static void endInstance() {
	delete _singleton;
    }
///
///\brief Constructeur privée/protected pour empécher l'instanciation n'importe où
///
protected:
    Singleton() {}
    virtual ~Singleton() {}

private:
    static T* _singleton;
};

template <class T>
T *Singleton<T>::_singleton = new T;

}

#endif

