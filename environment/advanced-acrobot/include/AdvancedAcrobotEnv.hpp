#ifndef ADVANCEDACROBOTENV_H
#define ADVANCEDACROBOTENV_H

#include "arch/AEnvironment.hpp"
#include "AdvancedAcrobotWorld.hpp"
#include "AdvancedAcrobotWorldView.hpp"
#include "bib/IniParser.hpp"

std::istream& operator>>(std::istream& istream, bone_joint& v);

class AdvancedAcrobotProgOptions {
public:
    static boost::program_options::options_description program_options() {
        boost::program_options::options_description desc("Allowed environment options");
        desc.add_options()
        ("view", "display the environment [default : false]");
        return desc;
    }
};

class AdvancedAcrobotEnv : public arch::AEnvironment<AdvancedAcrobotProgOptions>
{
public:
    AdvancedAcrobotEnv() {
        ODEFactory::getInstance();
    }

    ~AdvancedAcrobotEnv() {
        delete bones;
        delete actuators;
        ODEFactory::endInstance();
    }

    const std::vector<float>& perceptions() const
    {
        return instance->state();
    }

    float performance() {
        return instance->perf();
    }

    unsigned int number_of_actuators() const {
        return instance->activated_motors();
    }

private:
    void _unique_invoke(boost::property_tree::ptree* properties, boost::program_options::variables_map* vm) {
        if(vm->count("view"))
            visible = true;
        bones = bib::to_array<bone_joint>(properties->get<std::string>("environment.bones"));
        actuators = bib::to_array<bool>(properties->get<std::string>("environment.actuators"));

        if(visible)
            instance = new AdvancedAcrobotWorldView("data/textures");
        else
            instance = new AdvancedAcrobotWorld();
    }

    void _apply(const std::vector<float>& actuators) {
        instance->step(actuators);
    }

    void _next_instance() {
        instance->resetPositions();
    }

private:
    bool visible = false;
    std::vector<bone_joint>* bones;
    std::vector<bool>* actuators;
    AdvancedAcrobotWorld* instance;

    std::vector<float> internal_state;
};

#endif // ADVANCEDACROBOTENV_H
