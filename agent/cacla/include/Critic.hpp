#ifndef CRITIC_HPP
#define CRITIC_HPP

#include <bib/Combinaison.hpp>
#include <MLP.hpp>

#define DEBUG_FILE

template <typename Sample>
class Critic {
public:
    Critic(uint _nb_sensors, uint _nb_motors, bool _learnV, uint hidden_unit_v, bool lecun_activation, MLP* _actor, double _gamma) :
        nb_sensors(_nb_sensors), nb_motors(_nb_motors), actor(_actor), gamma(_gamma), learnV(_learnV) {

        if(_learnV) {
            if(hidden_unit_v == 0)
                vnn = new LinMLP(nb_sensors , 1, 0.0, lecun_activation);
            else
                vnn = new MLP(nb_sensors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);
        } else {//learn Q
            if(hidden_unit_v == 0)
                vnn = new LinMLP(nb_sensors + nb_motors , 1, 0.0, lecun_activation);
            else
                vnn = new MLP(nb_sensors + nb_motors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);
        }
    }

    ~Critic() {
        delete vnn;
    }

    double evaluateExploration(const Sample& sm) const {
        double exploration = sm.r;
        if (!sm.goal_reached) {
            if(learnV) {
                double nextV = vnn->computeOutVF(sm.next_s, {});
                exploration += gamma * nextV;
            } else {
                std::vector<double>* next_action = actor->computeOut(sm.next_s);
                double nextQ = vnn->computeOutVF(sm.next_s, *next_action);
                exploration += gamma * nextQ;
                delete next_action;
            }
        }

        return exploration;
    }

    double evaluateExploitation(const Sample& sm) const {
        if(learnV)
            return vnn->computeOutVF(sm.s, {});

        return vnn->computeOutVF(sm.s, sm.a);
    }

    double evaluate(const std::vector<double>& s, const std::vector<double>& a) const {
        if(learnV)
            return vnn->computeOutVF(s, {});

        return vnn->computeOutVF(s, a);
    }

    void write_critic_file(const std::string& file) {
#if defined(DEBUG_FILE) && !defined(NDEBUG)
        std::ofstream out;
        if(learnV)
            out.open("V." + file, std::ofstream::out);
        else
            out.open("Q." + file, std::ofstream::out);

        auto iter_V = [&](const std::vector<double>& x) {
            for(uint i=0; i < x.size(); i++)
                out << x[i] << " ";

            out << vnn->computeOutVF(x, {});
            out << std::endl;
        };

        auto iter_Q = [&](const std::vector<double>& x) {
            for(uint i=0; i < x.size(); i++)
                out << x[i] << " ";

            std::vector<double> m(nb_motors);
            for(uint i=nb_sensors; i < nb_motors + nb_sensors; i++)
                m[i - nb_sensors] = x[i];

            out << vnn->computeOutVF(x, m);
            out << std::endl;
        };

        if(learnV)
            bib::Combinaison::continuous<>(iter_V, nb_sensors, -1, 1, 50);
        else
            bib::Combinaison::continuous<>(iter_Q, nb_sensors+nb_motors, -1, 1, 50);

        out.close();
#endif
    }

    void forgotAll() {
        fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
    }

    double error() {
        return fann_get_MSE(vnn->getNeuralNet());
    }

    MLP* getMLP() {
        return vnn;
    }

    struct ParrallelTargetComputing {
        // Learn V
        ParrallelTargetComputing(const std::vector<Sample>& _vtraj, Critic* _ptr) :
            vtraj(_vtraj), ptr(_ptr) {

            data = fann_create_train(vtraj.size(), ptr->nb_sensors, 1);

            for (uint n = 0; n < vtraj.size(); n++) {
                Sample sm = vtraj[n];
                for (uint i = 0; i < ptr->nb_sensors ; i++)
                    data->input[n][i] = sm.s[i];
            }
        }

        // Learn Q
        ParrallelTargetComputing(const std::vector<Sample>& _vtraj, const std::vector<Sample>* _vtraj_encouraged, bool _encourage_absorbing, Critic* _ptr) :
            vtraj(_vtraj), vtraj_encouraged(_vtraj_encouraged), ptr(_ptr), actions(vtraj.size()) {

            if(_encourage_absorbing)
                data = fann_create_train(vtraj.size() + _vtraj_encouraged->size(), ptr->nb_sensors + ptr->nb_motors, 1);
            else
                data = fann_create_train(vtraj.size(), ptr->nb_sensors + ptr->nb_motors, 1);

            for (uint n = 0; n < vtraj.size(); n++) {
                Sample sm = vtraj[n];
                for (uint i = 0; i < ptr->nb_sensors ; i++)
                    data->input[n][i] = sm.s[i];
                
                for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
                    data->input[n][i] = sm.a[i - ptr->nb_sensors];

                actions[n] = ptr->actor->computeOut(sm.next_s);
            }

            for (uint n = 0; n < _vtraj_encouraged->size(); n++) {
                Sample sm = _vtraj_encouraged->at(n);
                for (uint i = 0; i < ptr->nb_sensors ; i++)
                    data->input[vtraj.size() + n][i] = sm.s[i];
                
                for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
                    data->input[vtraj.size() + n][i] = sm.a[i - ptr->nb_sensors];
            }
        }

        ~ParrallelTargetComputing() { //must be empty cause of tbb

        }

        void free() {
            fann_destroy_train(data);
            
            for(auto it : actions)
              delete it;
        }
        
        void update_actions(){
          for(auto it : actions)
            delete it;
          
          for (uint n = 0; n < vtraj.size(); n++){
            Sample sm = vtraj[n];
            actions[n] = ptr->actor->computeOut(sm.next_s);
          }
        }

        void operator()(const tbb::blocked_range<size_t>& range) const {

            struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());

            if(ptr->learnV){
              for (size_t n = range.begin(); n < range.end(); n++) {
                  Sample sm = vtraj[n];

                  double delta = sm.r;
                  if (!sm.goal_reached) {
                      double nextV = MLP::computeOutVF(local_nn, sm.next_s, {});
                      delta += ptr->gamma * nextV;
                  }

                  data->output[n][0] = delta;
              }
            } else {
              for (size_t n = range.begin(); n < range.end(); n++) {
                  Sample sm = n < vtraj.size() ? vtraj[n] : vtraj_encouraged->at(n - vtraj.size());

                  double delta = sm.r;
                  if (!sm.goal_reached) {
                      double nextV = MLP::computeOutVF(local_nn, sm.next_s, *actions[n]);
                      delta += ptr->gamma * nextV;
                  }

                  data->output[n][0] = delta;
              }
            }

            fann_destroy(local_nn);
        }

        struct fann_train_data* data;
        const std::vector<Sample>& vtraj;
        const std::vector<Sample>* vtraj_encouraged;
        const Critic* ptr;
        std::vector<std::vector<double>*> actions;
    };

    ParrallelTargetComputing* createTargetData(const std::vector<Sample>& _vtraj, const std::vector<Sample>* _vtraj_encouraged, bool encourage_absorbing) {
        if(learnV)
            return new ParrallelTargetComputing(_vtraj, this);

        return new ParrallelTargetComputing(_vtraj, _vtraj_encouraged, encourage_absorbing, this);
    }

protected:
    uint nb_sensors;
    uint nb_motors;
    MLP* vnn;
    MLP* actor;
    double gamma;
    bool learnV;
};

#endif
