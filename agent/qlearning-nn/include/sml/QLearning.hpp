#ifndef SARSANN_H
#define SARSANN_H

#include <set>
#include <vector>
#include <utility>
#include <deque>
#include <string>
#include <list>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "boost/graph/graph_concepts.hpp"
#include "doublefann.h"

#include "sml/Policy.hpp"
#include "bib/Utils.hpp"
#include "bib/Seed.hpp"
#include "sml/Q.hpp"

using std::pair;

namespace sml {

typedef struct fann* NN;

template <class State>
class QLearning : public Policy<State> {
  typedef struct _replay_history {
    State st;
    std::unique_ptr<DAction> at;
    State next_st;
    double next_r;
    bool end;
    int meeted;
  } replay_history;

  typedef struct _info_rep_list {
    double sum_reward;
    unsigned int replay_time;
  } info_replay;

  typedef boost::shared_ptr<replay_history> history_type;

  struct _replay_compare {
    bool operator()(const history_type a, const history_type b) const {
      for (int i = 0; i < a->st->size(); i++)
        if (a->st->at(i) < b->st->at(i)) return true;
      if (a->at->get(0) < b->at->get(0)) return true;
      for (int i = 0; i < a->next_st->size(); i++)
        if (a->next_st->at(i) < b->next_st->at(i)) return true;
      //         if different reward we want an average
      //             if(a->next_r < b->next_r)
      //                 return true;
      return false;
    }
  };

 public:
  QLearning(const ActionTemplate* atmp, RLParam param,
            unsigned int _size_input_state
#ifdef ACTION_TIME
            ,
            std::vector<int> _actions_time
#endif
           )
    : Policy<State>(param),
      atmpl(atmp),
      Qa(atmp),
      neural_networks(atmp->sizeNeeded()),
      size_input_state(_size_input_state),
      history_order(nullptr),
      internal_step(0)
#ifdef ACTION_TIME
    ,
      actions_time(_actions_time)
#endif
  {
    for (unsigned int i = 0; i < atmp->sizeNeeded(); i++) {
      neural_networks[i] =
        fann_create_standard(3, size_input_state, param.hidden_unit, 1);

      if (param.activation == "tanh") {
        fann_set_activation_function_hidden(neural_networks[i], FANN_SIGMOID_SYMMETRIC);
      } else if (param.activation == "sigmoid") {
        fann_set_activation_function_hidden(neural_networks[i], FANN_SIGMOID);
      } else if (param.activation == "linear") {
        fann_set_activation_function_hidden(neural_networks[i], FANN_LINEAR);
      } else {
        LOG_ERROR(
          "activation function for hidden layer of the neural network "
          "unknown : "
          << param.activation_stepness);
        exit(1);
      }
      fann_set_activation_steepness_hidden(neural_networks[i], param.activation_stepness);

      fann_set_activation_function_output(neural_networks[i], FANN_LINEAR);  // Linear cause Q(s,a) isn't normalized
      fann_set_learning_momentum(neural_networks[i], 0.);
      fann_set_train_error_function(neural_networks[i], FANN_ERRORFUNC_LINEAR);
      fann_set_training_algorithm(neural_networks[i], FANN_TRAIN_INCREMENTAL);
      fann_set_train_stop_function(neural_networks[i], FANN_STOPFUNC_MSE);
      fann_set_learning_rate(neural_networks[i], this->param.alpha);
      //      fann_randomize_weights(neural_networks[i], -0.0025, 0.0025);
    }
  }

  QLearning(const QLearning& clone)
    : Policy<State>(clone.param),
      atmpl(clone.atmpl),
      Qa(clone.Qa),
      neural_networks(clone.atmpl->sizeNeeded()),
      size_input_state(clone.size_input_state),
      history_order(nullptr),
      hoh_list(),
      lastState(),
      internal_step(clone.internal_step)
#ifdef ACTION_TIME
    ,
      actions_time(clone.actions_time)
#endif
  {
    for (unsigned int i = 0; i < clone.atmpl->sizeNeeded(); i++)
      neural_networks[i] = fann_copy(clone.neural_networks[i]);
  }

  ~QLearning() {
    for (unsigned int i = 0; i < atmpl->sizeNeeded(); i++)
      fann_destroy(neural_networks[i]);

    delete lastAction;

    for (auto bit = hoh_list.cbegin(); bit != hoh_list.cend(); ++bit)
      delete bit->first;

    if (history_order != nullptr) delete history_order;
  }

  void startEpisode(const State& s, const DAction& a) {
    if (lastAction != nullptr) delete lastAction;
    lastAction = new DAction(a);
    lastState = s;

    history_order = new std::list<history_type>;

    // start an new episod
    computeQa(s);
    internal_step = 0;
    // addTraces(s, a); // don't put the first trace, it may be not a good initilization
  }

  DAction* decision(const State& state, bool greedy) {
    if (greedy && bib::Utils::rand01(this->param.epsilon)) {
      return new DAction(atmpl, {RAND() % static_cast<int>(atmpl->sizeNeeded())});
    }

    computeQa(state);
    return Qa.argmax();
  }

  LearnReturn _learn(const State& state, double r, bool goal) {
    assert(this->lastAction != nullptr);
    DAction* a = this->lastAction;

    double delta = r;

    // For all a in A(s')
    computeQa(state);

    bool gotGreedy = false;
    DAction* ap = this->Qa.argmax();
    if (bib::Utils::rand01(this->param.epsilon)) {
      delete ap;
      ap = new DAction(this->atmpl, RAND() % this->atmpl->sizeNeeded());
      gotGreedy = true;
    }
#ifdef SARSA
    if (!goal) {
#ifdef ACTION_TIME
      delta = delta + pow(this->param.gamma, actions_time[a->hash()]) * this->Qa(*ap);
#else
      delta = delta + this->param.gamma * this->Qa(*ap);
#endif
    }
#else
    DAction* ba = this->Qa.argmax();

    if (!goal) {
#ifdef ACTION_TIME
      delta = delta +
              powf(this->param.gamma, actions_time[a->hash()]) * this->Qa(*ba);
#else
      delta = delta + this->param.gamma * this->Qa(*ba);
#endif
    }
    delete ba;
#endif

    if (internal_step > 0) {
      fann_type* inputs = new fann_type[size_input_state];
      fann_type out[1];

      for (unsigned int i = 0; i < size_input_state; i++)
        inputs[i] = lastState->at(i);
      out[0] = delta;

      fann_train(neural_networks[a->hash()], inputs, out);
      delete[] inputs;
    }

    if (internal_step > this->param.memory_size)
      this->addTraces(lastState, *a, state, r, goal);
    internal_step++;

    // take action a, observe reward, and next state
    delete this->lastAction;
    this->lastAction = ap;
    lastState = state;

    return {ap, gotGreedy};
  }

  Policy<State>* copyPolicy() {
    return new QLearning(*this);
  }

  void save(boost::archive::xml_oarchive*) {}

  void load(boost::archive::xml_iarchive*) {}

#ifndef NDEBUG
  float mse() {
    float s = 0.f;
    for (unsigned int i = 0; i < atmpl->sizeNeeded(); i++) {
      s += fann_get_MSE(neural_networks[i]);
      fann_reset_MSE(neural_networks[i]);
    }
    return s / atmpl->sizeNeeded();
  }
#endif


  void write(const string& chemin) {
    for (unsigned int i = 0; i < atmpl->sizeNeeded(); i++) {
      fann_save(neural_networks[i], (chemin + "." + std::to_string(i)).c_str());
    }
  }

  void read(const string& chemin) {
    for (unsigned int i = 0; i < atmpl->sizeNeeded(); i++) {
      neural_networks[i] =
        fann_create_from_file((chemin + "." + std::to_string(i)).c_str());
    }
  }

  int history_size() const {
    if (history_order != nullptr)
      return history_order->size();
    else return -1;
  }

  void resetTraces(double reward_sum) {
    if (history_order->size() > 0) {
      history_of_history p(history_order, {reward_sum, 0});
      hoh_list.push_back(p);
    } else {
      delete history_order;
      history_order = nullptr;
    }

    std::random_shuffle(hoh_list.begin(), hoh_list.end());
    replayTraces();
    if (hoh_list.size() > this->param.repeat_replay) {
      double min = reward_sum;
      auto it = hoh_list.begin();
      decltype(it) min_it;
      bool doNotDelete = false;
      for (; it != hoh_list.end() ;) {
        if (it->second.replay_time > (this->param.repeat_replay + 2) * 3) {  // too many replayed
          if (it->first == history_order)
            history_order = nullptr;
          delete it->first;
          it = hoh_list.erase(it);
          doNotDelete = true;
          break;
        }

        it->second.replay_time++;
        if (it->second.sum_reward <= min) {
          min = it->second.sum_reward;
          min_it = it;
        }
        ++it;
      }
      if (!doNotDelete) {
        if (min_it->first == history_order)
          history_order = nullptr;
        delete min_it->first;
        hoh_list.erase(min_it);
      }
    }
  }

#ifndef NDEBUG
  float weight_sum() {
    double sum = 0.f;
    for (unsigned int i = 0; i < atmpl->sizeNeeded(); i++) {
      struct fann_connection* connections = (struct fann_connection*) calloc(
                                              fann_get_total_connections(neural_networks[i]), sizeof(struct fann_connection));

      for (unsigned int j = 0; j < fann_get_total_connections(neural_networks[i]); j++)
        connections[j].weight = 0;

      fann_get_connection_array(neural_networks[i], connections);

      for (unsigned int j = 0; j < fann_get_total_connections(neural_networks[i]); j++)
        sum += std::fabs(connections[j].weight);

      free(connections);
    }


    return sum / atmpl->sizeNeeded();
  }
#endif


 protected:
  struct ParallelComputeQa {
    sml::QTable* Qa;
    const std::vector<sml::NN>& neural_networks;
    fann_type* inputs;
    ParallelComputeQa(sml::QTable* _Qa, const std::vector<sml::NN>& _neural_networks,
                      fann_type* _inputs)
      : Qa(_Qa), neural_networks(_neural_networks), inputs(_inputs) {}
    void operator()(const tbb::blocked_range<int>& r) const {
      for (int i = r.begin(); i != r.end(); i++) {
        fann_type* out = fann_run(neural_networks[i], inputs);
        Qa->operator()(0, i) = out[0];
      }
    }
  };

  void computeQa(const State& state) {
    fann_type* inputs = new fann_type[size_input_state];

    for (unsigned int i = 0; i < size_input_state; i++)
      inputs[i] = state->at(i);

    ParallelComputeQa para(&Qa, neural_networks, inputs);
    tbb::parallel_for(tbb::blocked_range<int>(0, atmpl->sizeNeeded()), para);
    delete[] inputs;
  }

  void addTraces(const State& state, const DAction& a, const State& next_st,
                 double next_r, bool end) {
    history_type new_play = history_type(new replay_history);
    new_play->st = state;
    new_play->at = std::unique_ptr<DAction>(new DAction(a));
    new_play->next_st = next_st;
    new_play->next_r = next_r;
    new_play->meeted = 1;
    new_play->end = end;

    history_order->push_back(new_play);
  }

  void replayTraces() {
    for (auto bit = hoh_list.cbegin(); bit != hoh_list.cend(); ++bit)
      if (bit->second.sum_reward != 0) {
        //         for(int n=0; n<this->param.repeat_replay; n++)
        for (typename std::list<history_type>::const_reverse_iterator it =
               bit->first->crbegin();
             it != bit->first->crend(); ++it) {
          const history_type play = *it;
          computeQa(play->next_st);
          DAction* ba = Qa.argmax();
#ifdef ACTION_TIME
          double delta = play->next_r;

          if (!play->end)
            delta += powf(this->param.gamma, actions_time[play->at->hash()]) *
                     this->Qa(*ba);
#else
          double delta = play->next_r;
          if (!play->end) delta += this->param.gamma * this->Qa(*ba);
#endif
          delete ba;

          fann_type* inputs = new fann_type[size_input_state];
          fann_type out[1];

          for (unsigned int i = 0; i < size_input_state; i++)
            inputs[i] = play->st->at(i);
          out[0] = delta;

          fann_train(neural_networks[play->at->hash()], inputs, out);
          delete[] inputs;
        }
      }
  }

 protected:
  const ActionTemplate* atmpl;
  QTable Qa;
  vector<NN> neural_networks;
  unsigned int size_input_state;

  std::list<history_type>* history_order;

  typedef std::pair<std::list<history_type> *, info_replay> history_of_history;
  std::deque<history_of_history> hoh_list;

  DAction* lastAction = nullptr;
  State lastState;
  int internal_step;

#ifdef ACTION_TIME
  std::vector<int> actions_time;
#endif
};
}  // namespace sml

#endif  // SARSANN_H
