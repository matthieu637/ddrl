#ifndef ACTIONTEMPLATE_HPP
#define ACTIONTEMPLATE_HPP

///
///\file ActionTemplate.hpp
///\brief le modèle pour un ensemble d'actions
///
///

#include <string>
#include <list>
#include <boost/unordered/unordered_map.hpp>
// #include <boost/serialization/nvp.hpp>
// #include <boost/serialization/list.hpp>

using std::list;
using std::string;
//using boost::serialization::make_nvp;

namespace sml {

    class ActionTemplate {
      public:
        typedef boost::unordered::unordered_map<string, int>::const_iterator nameIterator;
//     ActionTemplate();//empty constructor for serialization

///
///\brief Constructeur
///\param names : la liste des noms pour chaque action
///     sizes : le nombre des valeurs possibles pour chaque action
        ActionTemplate(const std::list<string>& names, const std::list<int>& sizes);

        ActionTemplate(const ActionTemplate&, const ActionTemplate&);
        ActionTemplate(const ActionTemplate&);

        ~ActionTemplate();

///
///\brief Retourner la position de l'action dans la liste
///\param name : le nom de l'action
        int indexFor(const string& name) const;

///
///\brief Retourner le nombre des actions différentes( accélérer + diriger = 2 )
        int actionNumber() const;


        int actionNumber(const string& name) const;
///
///\brief Retourner le nombre des valeurs possibles pour chaque action
        const std::list<int>* sizesActions() const;

///
///\brief Comparer deux modèles d'action
///\param ac : le modèle d'action à comparer
///\return True si deux modèles d'action sont les mêmes, False sinon
        bool operator==(const ActionTemplate& ac) const;

///
///\brief Retourner le taille necéssaire pour toutes les actions
        unsigned int sizeNeeded() const;

        const boost::unordered_map< string, int>* getActionNames() const;

        void setSize(const string& s, int val);

        /*
        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
          (void) version;
          ar & make_nvp("ActionName", actionNames);
          ar & make_nvp("Sizes", sizes);

        }*/
      private :
        boost::unordered_map< string, int> actionNames;
        std::list<int> sizes;
    };

    typedef ActionTemplate StateTemplate;

}

#endif // ACTIONTEMPLATE_HPP
