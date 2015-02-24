
#ifndef ACTION_HPP
#define ACTION_HPP

///
///\file Action.hpp
///\brief Stocker des actions dynamiques et discrétisées
///
///

#include <string>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include "sml/ActionTemplate.hpp"



using std::string;
using boost::serialization::make_nvp;

namespace sml {

    class DAction {

      public:

//friend class boost::serialization::access;


///
///\brief Constructeur
///\param temp : le modèle de la nouvelle action
///     vals : les valeurs des actions
///
        DAction(const ActionTemplate* temp, const std::list<int>& vals);

///
///\brief Constructeur
///\param temp : le modèle de la nouvelle action
///     value : le hash des actions
///
        DAction(const ActionTemplate* temp, int value);

        DAction(const DAction& a);
        ~DAction();

///
///\brief Retourner la valeur d'une action
///\param index : le numéro de l'action
        int get( int index) const;

///
///\brief Retourner la valeur d'une action
///\param name : le nom de l'action
        int get(const string& name) const;
        int operator[](const string& name) const;

///
///\brief Modifier la valeur de l'action
///\param name : le nom de l'action
///       value : la valeur de l'action
        void set(const string& name, int value);

        void copyValuesOf(const DAction& ac);

///
///\brief Comparer deux actions
///\param ac : l'action à comparer
///\return True si deux actions sont les mêmes, False sinon
        bool operator==(const DAction& ac) const;

///
///\brief Ordonner deux actions
///\param ac : l'action à comparer
///\return True si le hash de "ac" est plus grand, False sinon
        bool operator<(const DAction& ac) const;

///
///\brief Retourner le hach de l'action
        unsigned int hash() const;

        void print(std::ostream &flux) const;


//     BOOST_SERIALIZATION_SPLIT_MEMBER()
//     template<class Archive>
//     void save(Archive& ar, const unsigned int version) const {
//         (void) version;
//         ar << make_nvp("Template", this->templ);
//
//         for(int i = 0 ; i< templ->actionNumber(); i++)
//             ar << make_nvp("values", values[i]);
//     }
//
//     template<class Archive>
//     void load(Archive& ar, const unsigned int version) {
//         (void) version;
//         ar >> make_nvp("Template", this->templ);
//
//         values = new int[this->templ->actionNumber()];
//         for(int i = 0 ; i< templ->actionNumber(); i++)
//             ar >> make_nvp("values", values[i]);
//     }

      private:

///
///\brief Calculer le hash de l'action
        void computehash();

        int *values = nullptr;
        const ActionTemplate *templ;
        int hashmem;
    };


///
///\brief Pour afficher l'action dans un flux
    std::ostream& operator<< (std::ostream& stream, const sml::DAction& ac);

    typedef DAction DState;

}

#endif
