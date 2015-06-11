#include "PowerAg.hpp"

using namespace std;
using namespace Eigen;

PowerAg::PowerAg(unsigned int nb_motors, unsigned int nb_sensors) : n_sensors(nb_sensors), n_motors(nb_motors) {
      n_dims = nb_sensors/2;
      y_max = nb_sensors/2;
      best_value=-y_max;
      iter = 0;
      episode = 0;
      best_reward = 0;
  }

const vector<float>& PowerAg::run(float _reward, const vector<float>& sensors, bool learning, bool goal_reached) {

    if(pas<0){
        vector<float> actions;
        for(unsigned int i=0;i<n_motors;i++){
            actuator.push_back(0.f);
        }
    } else {
        actuator = algo->getNextActions(sensors);
    }

      float theta(0);
      float y(0);
      for(unsigned int dim=0; dim<n_dims;dim++){
        theta +=sensors[dim*2];
        y+= -cos(theta);
      }
      if(y>best_value)
        best_value=y;
      /*if(goal_reached){
            int diff = pas-config.n_steps_max;
            rewards[iter] += 0.1f*abs(diff)/config.n_steps_max;
      }*/

      if(y>0.99f*y_max && sensors[1]<0.1){
          rewards[iter] += .1f*y/y_max;
     }

      for(unsigned int i=0;i<n_sensors;i++)
        state(pas,i) = sensors[i];
      for(unsigned int j=0;j<n_motors;j++)
        state(pas,n_sensors+j) = actuator[j];

      pas++;

    return actuator;
  }

  void PowerAg::start_episode(const std::vector<float>& sensors) {
    pas=0;
    rewards.push_back(0);
    algo->computeNewWeights();
    if(iter%config.n_instances==0)
        best_reward_episode=0;
  }

  void PowerAg::end_episode() {
    rewards[iter]=(best_value+y_max)/(2*y_max)+rewards[iter];
    best_value=-y_max;
    reward =rewards[iter];
    if(reward>best_reward){
        best_state = state;
        best_reward=reward;
    }
    if(reward>best_reward_episode){
        best_reward_episode=reward;
    }

    algo->addReward(rewards[iter]);
    iter++;
    episode = iter/config.n_instances;
  }

   void PowerAg::save(const std::string& path) {

        algo->save(path);
       const string file_best_state="best_state.mat";
       ofstream fichier_state(file_best_state, ios::out | ios::trunc);
        if(fichier_state)
        {

        for(unsigned int i=0;i<config.n_steps_max;i++){
            for(unsigned int state=0;state<n_sensors+n_motors;state++){
                if(state!=n_sensors+n_motors-1)
                    fichier_state << best_state(i,state) << "\t";
                else
                    fichier_state << best_state(i,state) << endl;
            }
        }
        fichier_state.close();
        }
        else
            cerr << "Impossible d'ouvrir le fichier !" << endl;

  }

  void PowerAg::load(const std::string& path) {
    algo->load(path);
  }

