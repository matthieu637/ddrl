#include "sml/ActionTemplate.hpp"

#include <assert.h>
#include <bib/Logger.hpp>

namespace sml {

    ActionTemplate::ActionTemplate(const sml::ActionTemplate& a, const sml::ActionTemplate& b): actionNames(a.actionNames.size() + b.actionNames.size()), sizes(a.sizes) {
        for (boost::unordered_map< string, int>::const_iterator it = a.actionNames.cbegin(); it != a.actionNames.cend(); ++it) {
            actionNames[it->first] = it->second;
        }
        for (boost::unordered_map< string, int>::const_iterator it = b.actionNames.cbegin(); it != b.actionNames.cend(); ++it) {
            actionNames[it->first] = it->second + a.actionNames.size();
        }
        for (std::list<int>::const_iterator it = b.sizes.cbegin(); it != b.sizes.cend(); ++it) {
            sizes.push_back(*it);
        }

//     bib::Logger::PRINT_ELEMENTS<list<int>>(sizes);
//     for(boost::unordered_map< string, int>::const_iterator it= actionNames.cbegin(); it != actionNames.cend(); ++it) {
//  LOG_DEBUG(it->first << " " << it->second);
//     }
    }

    ActionTemplate::ActionTemplate(const sml::ActionTemplate& a): actionNames(a.actionNames.size()), sizes(a.sizes) {
        for (boost::unordered_map< string, int>::const_iterator it = a.actionNames.cbegin(); it != a.actionNames.cend(); ++it) {
            actionNames[it->first] = it->second;
        }
    }

    ActionTemplate::ActionTemplate(const std::list<string>& names, const std::list<int>& sizes): actionNames(names.size()), sizes(sizes) {
        assert(names.size() == sizes.size());

        unsigned int i = 0;
        for (std::list<string>::const_iterator it = names.begin(); it != names.end(); ++it) {
            actionNames[*it] = i;
            i++;
        }
    }

    ActionTemplate::~ActionTemplate() {
        actionNames.clear();
        sizes.clear();
    }

    int ActionTemplate::actionNumber(const string& name) const {
        unsigned int reach = actionNames.at(name);
        unsigned int i = 0;

        std::list<int>::const_iterator it;
        for (it = sizes.begin(); it != sizes.end(); ++it) {
            if (i == reach)
                return *(it);
            i++;
        }

        return (*it);
    }

    int ActionTemplate::indexFor(const string& name) const {
//     LOG_DEBUG(name << " " << actionNames.size());
        assert(name.size() > 0 && actionNames.find(name) != actionNames.end());
        return actionNames.at(name);
    }

    int ActionTemplate::actionNumber() const {
        return actionNames.size();
    }


    bool ActionTemplate::operator==(const ActionTemplate& ac) const {
        return actionNames == ac.actionNames && sizes == ac.sizes;
    }

    const std::list<int>* ActionTemplate::sizesActions() const {
        return &this->sizes;
    }

    const boost::unordered_map< string, int>* ActionTemplate::getActionNames() const {
        return &actionNames;
    }

    void ActionTemplate::setSize(const string& s, int val) {
        int index = indexFor(s);
        int i = 0;
        for (list<int>::iterator it = sizes.begin(); it != sizes.end(); ++it) {
            if (i == index)
                *it = val;
            i++;
        }
    }

    unsigned int ActionTemplate::sizeNeeded() const {
        unsigned int r = 1;

        for (list<int>::const_iterator it = sizes.cbegin(); it != sizes.cend(); ++it)
            r *= (*it);
        return r;
    }

}
