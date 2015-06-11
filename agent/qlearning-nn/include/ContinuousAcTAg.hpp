#ifndef CONTINUOUSACTAG_HPP
#define CONTINUOUSACTAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "arch/AAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "MLP.hpp"

#define NBSOL_OPT 25
#define LEARNING_PRECISION 0.00001
#define MIN_Q_ITERATION 10
#define MAX_Q_ITERATION 100
#define MAX_NEURAL_ITERATION 50000

typedef struct _sample {
  std::vector<float> s;
  std::vector<float> a;
  std::vector<float> next_s;
  double r;
  bool goal_reached;
  double score;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(a);
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
    ar& BOOST_SERIALIZATION_NVP(score);
  }

  bool operator< (const _sample& b) const {
    if(score < b.score)
        return true;
    for (uint i = 0; i < s.size(); i++)
      if (s[i] < b.s[i])
        return true;
    for (uint i = 0; i < a.size(); i++)
      if (a[i] < b.a[i])
        return true;
    for (uint i = 0; i < next_s.size(); i++)
      if (next_s[i] < b.next_s[i])
        return true;

    if (r < b.r)
      return true;

    return goal_reached < b.goal_reached;
  }

} sample;

class ContinuousAcTAg : public arch::AAgent<> {
 public:
  ContinuousAcTAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~ContinuousAcTAg() {
    delete nn;
  }
  
  float max_reward = std::numeric_limits<float>::min();
  
  const std::vector<float>& runf(float r, const std::vector<float>& sensors,
                                bool learning, bool goal_reached, bool last_step) override {

    double reward = r;
    if (reward >= 1.f) {
      reward = 13000;

//             uint keeped = 2000 - internal_time;
//             reward = 100 * log2(keeped + 2);
    } else
      reward = exp((reward - 0.01)*4000) * 0.01;
    
//     if(max_reward < reward)
//       max_reward = reward;
//                                   
//     if(!goal_reached && !last_step)
//       reward = 0;
//     else
//       reward = max_reward;
//     if(!goal_reached)
//       reward = 0;
    
    internal_time ++;

    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;

    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;

    time_for_ac--;
    if (time_for_ac == 0 || goal_reached) {
      const std::vector<float>& next_action = _run(weighted_reward, sensors, learning, goal_reached);
//       time_for_ac = bib::Utils::transform(next_action[nb_motors], -1., 1., min_ac_time, max_ac_time);
      time_for_ac = 5;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      weighted_reward = 0;
      pow_gamma = 1.f;
    }

    return returned_ac;
  }

  const std::vector<float>& _run(float reward, const std::vector<float>& sensors,
                                 bool learning, bool goal_reached) {
    vector<float>* next_action = nullptr;

    if (softmax) {
      struct NNDistribution {
        MLP* mlp;
        const std::vector<float>& sensors;
        NNDistribution(MLP* _mlp, const std::vector<float>& _sensors): mlp(_mlp), sensors(_sensors) {}

        float eval(const std::vector<float>& x) {
          return mlp->computeOut(sensors, x);
        }
      };
      NNDistribution dist(nn, sensors);
      bib::MCMC<NNDistribution, float> mcmc(&dist);
      static std::vector<float> xinit(2, 0);
      next_action = new vector<float>(*mcmc.oneStep(0.35, xinit, 5).get());
    } else {
      if (init_old_ac && last_action.get() != nullptr)
        next_action = nn->optimized(sensors, *last_action, NBSOL_OPT);
      else
        next_action = nn->optimized(sensors, {}, NBSOL_OPT);
    }

    if (last_action.get() != nullptr && learning) {  // Update Q
//       double nextQ = nn->computeOut(sensors, *next_action);
//       if (!goal_reached){
//         if(aware_ac_time)
//           nn->learn(last_state, *last_action, reward + pow(gamma, bib::Utils::transform(last_action->at(last_action->size()-1),-1.,1., min_ac_time, max_ac_time) ) * nextQ);
//         else
//           nn->learn(last_state, *last_action, reward + gamma * nextQ);
//       }
//       else{
//         nn->learn(last_state, *last_action, reward);
//       }
//             trajectory.push_back( {last_state, *last_action, sensors, reward});
      auto ppair = trajectory.insert( {last_state, *last_action, sensors, reward, goal_reached, (uint)0});
      if(ppair.second)
        current_trajectory.push_back(*ppair.first);
    }

    if (!softmax && learning && bib::Utils::rand01() < alpha) {
      for (uint i = 0; i < next_action->size(); i++)
        next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
    }
    last_action.reset(next_action);

    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    return *next_action;
  }

  void _unique_invoke(boost::property_tree::ptree*, boost::program_options::variables_map*) override {
//         epsilon             = pt->get<float>("agent.epsilon");
//         gamma               = pt->get<float>("agent.gamma");
//         alpha               = pt->get<float>("agent.alpha");
//         hidden_unit         = pt->get<int>("agent.hidden_unit");
// //     rlparam->activation          = pt->get<std::string>("agent.activation_function_hidden");
// //     rlparam->activation_stepness = pt->get<float>("agent.activation_steepness_hidden");
// //
// //     rlparam->repeat_replay = pt->get<int>("agent.replay");
// //
// //     int action_per_motor   = pt->get<int>("agent.action_per_motor");
// //
// //     sml::ActionFactory::getInstance()->gridAction(nb_motors, action_per_motor);
// //     actions = new sml::list_tlaction(sml::ActionFactory::getInstance()->getActions());
// //
// //     act_templ = new sml::ActionTemplate( {"effectors"}, {sml::ActionFactory::getInstance()->getActionsNumber()});
// //     ainit = new sml::DAction(act_templ, {0});
// //     algo = new sml::QLearning<EnvState>(act_templ, *rlparam, nb_sensors);
    hidden_unit = 15;
    gamma = 0.99; // < 0.99  => gamma ^ 2000 = 0 && gamma != 1 -> better to reach the goal at the very end
//     gamma = 1.0d;
    //check 0,0099×((1−0.95^1999)÷(1−0.95))
    //r_max_no_goal×((1−gamma^1999)÷(1−gamma)) < r_max_goal * gamma^2000 && gamma^2000 != 0
    alpha = 0.05;
    epsilon = 0.2;

    min_ac_time = 4;
    max_ac_time = 4;

    aware_ac_time = false;
    init_old_ac = false;
    softmax = false;

//     nn = new MLP(nb_sensors + nb_motors + 1, hidden_unit, nb_sensors, alpha);
    nn = new MLP(nb_sensors + nb_motors, hidden_unit, nb_sensors, alpha);
    if (boost::filesystem::exists("trajectory.data")) {
      decltype(trajectory)* obj = bib::XMLEngine::load<decltype(trajectory)>("trajectory", "trajectory.data");
      trajectory = *obj;
      delete obj;

      end_episode();
    }
  }

  void start_episode(const std::vector<float>& sensors) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;

    weighted_reward = 0;
    pow_gamma = 1.d;
    time_for_ac = 1;
    sum_weighted_reward = 0;
    global_pow_gamma = 1.f;
    internal_time = 0;
//     trajectory.clear();
    current_trajectory.clear();
    max_reward = std::numeric_limits<float>::min();
  }

  struct ParrallelLearnFromScratch {
    ParrallelLearnFromScratch(struct fann_train_data* _data, const NN _model_nn, uint number_par)
      : data(_data), model_nn(_model_nn) {
      mses = new vector<double>(number_par);
      local_nn = new vector<NN>(number_par);
    }


    ~ParrallelLearnFromScratch() { //must be empty cause of tbb

    }

    NN bestNN() {
      double imin = 0;
      for (uint i = 1; i < local_nn->size(); i++)
        if (mses->at(imin) > mses->at(i))
          imin = i;

//             bib::Logger::PRINT_ELEMENTS<>(*mses);

      return local_nn->at(imin);
    }

    void free() {
      delete mses;
      for (uint i = 0; i < local_nn->size(); i++)
        fann_destroy(local_nn->at(i));
      delete local_nn;
    }

    void operator()(const tbb::blocked_range<uint>& range) const {
      for (uint index = range.begin(); index  < range.end(); index++) {
        local_nn->at(index) = fann_copy(model_nn);
        fann_randomize_weights(local_nn->at(index), -0.025, 0.025);
//         MLP::learn(local_nn->at(index), data);
        mses->at(index) = fann_get_MSE(local_nn->at(index));
      }
    }

    struct fann_train_data* data;
    const NN model_nn;
    vector<double>* mses;
    vector<NN>* local_nn;
  };

  struct DQtoQNext {
    DQtoQNext(const std::vector<sample>& _vtraj, const ContinuousAcTAg* _ptr) : vtraj(_vtraj), ptr(_ptr) {
//       data = fann_create_train(vtraj.size(), ptr->nb_sensors + ptr->nb_motors + 1, 1);
      data = fann_create_train(vtraj.size(), ptr->nb_sensors + ptr->nb_motors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
//         for (uint i = 0; i < ptr->nb_motors + 1  ; i++)
        for (uint i = 0; i < ptr->nb_motors  ; i++)
          data->input[n][ptr->nb_sensors + i] = sm.a[i];
      }
    }

    ~DQtoQNext() { //must be empty cause of tbb

    }

    void free() {
      fann_destroy_train(data);
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->nn->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        sample sm = vtraj[n];

        double delta = sm.r;
        if (!sm.goal_reached) {
          std::vector<float>* best_action = ptr->nn->optimized(sm.next_s, {}, NBSOL_OPT);
          double nextQ = MLP::computeOut(local_nn, sm.next_s, *best_action);
          if (ptr->aware_ac_time)
            delta += pow(ptr->gamma, bib::Utils::transform(sm.a[ptr->nb_motors + ptr->nb_sensors], -1., 1., ptr->min_ac_time,
                         ptr->max_ac_time)) * nextQ;
          else
            delta += ptr->gamma * nextQ;
          delete best_action;
        }

        data->output[n][0] = delta;
      }

      fann_destroy(local_nn);
    }

    struct fann_train_data* data;
    const std::vector<sample>& vtraj;
    const ContinuousAcTAg* ptr;
  };

  void end_episode() override {
    if (trajectory.size() > 0) {
      std::vector<sample> vtraj(trajectory.size());
      std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());

      DQtoQNext dq(vtraj, this);

      auto iter = [&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), dq);

//         fann_randomize_weights(nn->getNeuralNet(), -0.025, 0.025);
        nn->learn(dq.data, MAX_NEURAL_ITERATION, 0, LEARNING_PRECISION);

// not updated
//                 uint number_par = 6;
//                 ParrallelLearnFromScratch plfs(dq.data, nn->getNeuralNet(), number_par);
//                 tbb::parallel_for(tbb::blocked_range<uint>(0, number_par), plfs);
//                 nn->copy(plfs.bestNN());
//                 plfs.free();
      };

      auto eval = [&]() {
        return fann_get_MSE(nn->getNeuralNet());
      };

//             bib::Converger::determinist<>(iter, eval, 250, 0.0005, 1);

      NN best_nn = nullptr;
      auto save_best = [&]() {
        best_nn = fann_copy(nn->getNeuralNet());
      };

      bib::Converger::min_stochastic<>(iter, eval, save_best, MAX_Q_ITERATION, LEARNING_PRECISION * 10, 0, MIN_Q_ITERATION);
      nn->copy(best_nn);
      fann_destroy(best_nn);

      dq.free();
//       LOG_DEBUG("number of data " << trajectory.size());
      
      //puring by scoring
      for(auto it = current_trajectory.begin(); it != current_trajectory.end() ; it++){
          trajectory.erase(*it);
          it->score = sum_weighted_reward;
      }
      
      for(auto it = current_trajectory.begin(); it != current_trajectory.end() ; it++){
          trajectory.insert(*it);
      }

      while(trajectory.size() > 2000){
          trajectory.erase(trajectory.begin());
      }
          
    }
  }
  
  double sum_score_replayed() const {
      double sum= 0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; it++){
          sum += it->score;
      } 
      return sum;
  }

  void save(const std::string& path) override {
    nn->save(path);
    bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    nn->load(path);
  }

 protected:
  void _display(std::ostream& out) const override {
    out << sum_weighted_reward << " " << std::setw(8) << std::fixed << std::setprecision(5) << 
    nn->error() << " " << trajectory.size() << " " << std::setw(25) << std::fixed << std::setprecision(22) << sum_score_replayed() ;
  }
  
  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) << 
    sum_weighted_reward << " " << std::setw(8) << std::fixed << 
    std::setprecision(5) << nn->error() << " " << trajectory.size() ;
  }

 private:
  uint nb_motors;
  uint nb_sensors;
  uint time_for_ac;

  double weighted_reward;
  double pow_gamma;
  double global_pow_gamma;
  double sum_weighted_reward;

  uint min_ac_time;
  uint max_ac_time;

  uint internal_time;

  bool aware_ac_time;
  bool init_old_ac;

  bool softmax;

  double epsilon, alpha, gamma;
  uint hidden_unit;

  std::shared_ptr<std::vector<float>> last_action;
  std::vector<float> last_state;

  std::vector<float> returned_ac;

  std::set<sample> trajectory;
  std::list<sample> current_trajectory;
//     std::list<sample> trajectory;

  MLP* nn;
};

#endif

