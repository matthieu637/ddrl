#ifndef ADVANCEDACROBOTWORLD_HPP
#define ADVANCEDACROBOTWORLD_HPP
#include <vector>
#include <ode/ode.h>
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define BONE_LENGTH 0.3
#define BONE_LARGER 0.02
#define STARTING_Z 0.5

#define GRAVITY -9.81
#define BONE_DENSITY 1
#define BONE_MASS 1
#define MAX_TORQUE_HINGE 10
#define MAX_TORQUE_SLIDER 60
#define WORLD_STEP 0.01

enum bone_joint {HINGE, SLIDER};

class AdvancedAcrobotWorld
{
public:
    AdvancedAcrobotWorld(const std::vector<bone_joint>& types={HINGE,HINGE}, const std::vector<bool>& actuators={false,true,true});
    virtual ~AdvancedAcrobotWorld();

    void resetPositions();
    
    virtual void step(const std::vector<float>& motors);
    const std::vector<float>& state() const;
    bool end();
    bool prematureEnd();
    unsigned int activated_motors();
    float perf() const;
    
protected:
    void createWorld(const std::vector<bone_joint>&);
    std::vector<float>* current_joint_forces() const;

public:
    ODEWorld odeworld;
protected:
    std::vector<bone_joint> types;
    std::vector<bool> actuators;
    
    std::vector<ODEObject*> bones;
    std::vector<dJointID> joints;
    
    std::vector<float> internal_state;

    dGeomID ground;

private:
    bool goalBeenReached;
    bool goalFailed;
    unsigned int _activated_motors;
};

struct nearCallbackData {
    AdvancedAcrobotWorld* inst;
};

#endif // ADVANCEDACROBOTWORLD_HPP
