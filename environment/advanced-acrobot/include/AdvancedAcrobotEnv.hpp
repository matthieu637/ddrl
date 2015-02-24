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

class ProblemDefinition {
protected:
    virtual float performance(AdvancedAcrobotWorld*) = 0;
    virtual bool still_running(AdvancedAcrobotWorld*) {
        return true;
    }
};

class KeepHigh : ProblemDefinition
{
protected:
    float performance(AdvancedAcrobotWorld* instance) {
        return instance->perf();
    }
};

class ReachLimitPoorInformed : ProblemDefinition
{
protected:
    float performance(AdvancedAcrobotWorld* instance) {
        if(instance->perf() > 0.95)
            return 1.;
        else return 0.f;
    }

    bool still_running(AdvancedAcrobotWorld* instance) {
        return instance->perf() <= 0.95;
    }
};

class ReachLimitWellInformed : ProblemDefinition
{
protected:
    float performance(AdvancedAcrobotWorld* instance) {
        if(instance->perf() > 0.95)
            return 1.;
        else return instance->perf()*0.01;
    }
    
    bool still_running(AdvancedAcrobotWorld* instance) {
        return instance->perf() <= 0.95;
    }
};


template<typename ProblemToResolve=ReachLimitWellInformed>
class AdvancedAcrobotEnv : public arch::AEnvironment<AdvancedAcrobotProgOptions>, private ProblemToResolve
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
        return ProblemToResolve::performance(instance);
    }

    unsigned int number_of_actuators() const {
        return instance->activated_motors();
    }

    unsigned int number_of_sensors() const {
        return instance->state().size();
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

    bool _running() {
        return ProblemToResolve::still_running(instance);
    }

private:
    bool visible = false;
    std::vector<bone_joint>* bones;
    std::vector<bool>* actuators;
    AdvancedAcrobotWorld* instance;

    std::vector<float> internal_state;
};

#endif // ADVANCEDACROBOTENV_H
