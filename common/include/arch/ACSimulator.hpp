#ifndef ACSIMULATOR_HPP
#define ACSIMULATOR_HPP

#include <sys/wait.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include "tbb/parallel_for.h"
#include "bib/Thread.hpp"
#include "Simulator.hpp"
#include "AACAgent.hpp"

using std::string;

//#define VALGRIND

namespace arch {

template <typename Environment, typename Agent, typename Stat=DummyEpisodeStat>
class ACSimulator : public Simulator<Environment, Agent, Stat> {

  typedef Policy< typename Agent::PolicyImpl> _Policy;

 public:
  ACSimulator() : arch::Simulator<Environment, Agent, Stat>() {
        LOG_INFO("actor-critic simulator should only be used for policy/value function analyse");
        LOG_INFO("hypothesis : policy doesn't change online, determinist environment (possible to have stochasticity at initial state)");
  }

  ~ACSimulator() {
    
  }

  void enable_analyse_distance_bestVF(){
      if(this->properties == nullptr){
          LOG_ERROR("call enable_analyse_distance_bestVF after init");
          exit(1);
      }
    
      analyse_distance_bestVF = true;
      shm_id = (uint) (bib::Utils::rand01()*5000);
      
      boost::property_tree::ptree* p = this->properties;
      precision = p->get<unsigned int>("simulation.vbest_precision");
  }

protected:
    void run_episode(bool learning, unsigned int lepisode, unsigned int tepisode) override {
    this->env->reset_episode();
    std::list<double> all_rewards;
    this->agent->start_instance(learning);
    
    uint instance = 0;
    while (this->env->hasInstance()) {
      uint step = 0;
      Stat stat;
      std::vector<double> perceptions = this->env->perceptions();
      this->agent->start_episode(perceptions, learning);
      
      std::list< std::vector<double> > all_perceptions;
      std::list< std::vector<double> > all_actions;
      std::list< double > all_Vs;
      _Policy* policy = this->agent->getCopyCurrentPolicy();

      while (this->env->running()) {
        perceptions = this->env->perceptions();
        all_perceptions.push_back(perceptions);
        if(this->agent->did_decision())
          all_Vs.push_back(this->agent->criticEval(perceptions));
        double reward = this->env->performance();
        const std::vector<double>& actuators = this->agent->runf(reward, perceptions, learning, false, false);
        all_actions.push_back(actuators);
        this->env->apply(actuators);
        stat.dump(lepisode, perceptions, actuators, reward);
        all_rewards.push_back(reward);
        step++;
      }
      
      // if the environment is in a final state
      //        i.e it didn't reach the number of step but finished well
      // then we call the algorithm a last time to give him this information
      perceptions = this->env->perceptions();
      double reward = this->env->performance();
      this->agent->runf(reward, perceptions, learning, this->env->final_state(), true);
      all_rewards.push_back(reward);

      if(analyse_distance_bestVF){
        double diff = compareBestValueFonction(all_perceptions, all_actions, policy, this->agent->getGamma(), 
                                               all_Vs, this->agent->sum_weighted_rewards(), this->env->get_first_state_stoch());
        //double diff = 0;
#ifndef NDEBUG
        LOG_DEBUG("diff (higher bad) " << diff);
#endif
        LOG_FILE(std::to_string(instance)+".distanceBestVF.data", std::setprecision(12) << diff);
      }

      this->env->next_instance();
      this->agent->end_episode();
      
      delete policy;
      this->dump_and_display(lepisode, instance, tepisode, all_rewards, this->env, this->agent, learning, step);
      instance ++;
    }
    
    this->agent->end_instance(learning);
    
    if(learning)
      this->save_agent(this->agent, lepisode);
  }
  
  double compareBestValueFonction(std::list< std::vector<double> > all_perceptions, 
                                          std::list< std::vector<double> > all_actions, _Policy* policy, double gamma,
                                          std::list< double > all_Vs, double swr, const std::vector<double>& stochasticity
                                 ){
    
    double diff = 0;
    
    using namespace boost::interprocess;
    typedef allocator<double, managed_shared_memory::segment_manager>  ShmemAllocator;
    typedef boost::interprocess::vector<double, ShmemAllocator> MyVector;
    
    std::string sshm_filename ="ac_sim_forked_";
    sshm_filename += std::to_string(shm_id);
    const char* shm_filename = sshm_filename.c_str();
    
    shared_memory_object::remove(shm_filename);
    
    managed_shared_memory shm_obj(create_only, shm_filename, 65536 );
    const ShmemAllocator alloc_inst (shm_obj.get_segment_manager());
    
    MyVector *myvector = shm_obj.construct<MyVector>("result")(alloc_inst);
    
    for(uint i = 0; i < all_Vs.size(); ++i)
      myvector->push_back(0);
   
    std::vector<int> childs_pid;
    
    uint threads = bib::ThreadTBB::getInstance()->get_number_thread();
    
#if defined(DEBUG_WITH_DETERMINIST_POLICY) || defined(VALGRIND)
    threads=1;
#endif
    
    uint work_per_process = all_Vs.size() / threads;
    uint additional_work = all_Vs.size() - (work_per_process * threads);
    
    for (uint i = 0; i < threads; i++){
#ifdef VALGRIND
      int pid = 0;
#else
      int pid = fork();
#endif
      if(pid ==0){
        //Open the managed segment
        managed_shared_memory segment2(open_only, shm_filename);

        //Find the vector using the c-string name
        MyVector *myvector2 = segment2.find<MyVector>("result").first;
        
        Environment *lenv = new Environment;
        lenv->unique_invoke(this->properties, this->command_args);
        
        _Policy* lpol =  new _Policy(*policy);
        for(uint j=0;j < work_per_process; j++){
          uint z = (i*work_per_process) + j;
          myvector2->at(z) = evalBestValueFonction(all_actions, lpol, lenv, gamma, z, all_perceptions, swr, stochasticity);
        }
        
        if(i == 0){
          for(uint j=0;j < additional_work; j++){
            uint z = (bib::ThreadTBB::getInstance()->get_number_thread()*work_per_process) + j;
            myvector2->at(z) = evalBestValueFonction(all_actions, lpol, lenv, gamma, z, all_perceptions, swr, stochasticity);
          }
        }
        
        delete lpol;
        delete lenv;

#ifndef VALGRIND 
        exit(0);
#endif
      } else {
       childs_pid.push_back(pid);
      }
    }
    
    int status = 0;
    for(auto it : childs_pid){
      waitpid(it, &status, 0);
      if(!WIFEXITED(status) || WEXITSTATUS(status) != 0){
        LOG_ERROR("ERROR in sub process");
        exit(1);
      }
    }

    int i=0;
    for(auto it_vs : all_Vs){
      diff += (myvector->at(i)-it_vs)*(myvector->at(i)-it_vs);
      i++;
      ++it_vs;
    }
    
    shared_memory_object::remove(shm_filename);
    
    return diff / all_perceptions.size();
  }
  
  double evalBestValueFonction(std::list< std::vector<double> > all_actions, _Policy* lpol, Environment* lenv, 
                               double gamma, uint begin, std::list< std::vector<double> > all_perceptions, double swr, const std::vector<double>& stochasticity){

#ifndef DEBUG_WITH_DETERMINIST_POLICY
    (void) all_perceptions;
    (void) swr;
#endif
    
    std::vector<double>* mmean = new std::vector<double>(precision);
    
    for (uint mean = 0; mean < precision; mean++) {
      lenv->reset_episode_choose(stochasticity);
      lpol->reset_decision_interval();
      
      auto it = all_actions.cbegin();
      double local_gamma = 1.00000000000000f;
      double rsum = 0.00000000000000f;
      uint i = 0;
      std::list<double> inter_rewards;
      
#ifdef DEBUG_WITH_DETERMINIST_POLICY
      LOG_DEBUG("#########################################"<<mean);
      LOG_DEBUG("#########################################"<<begin);
      LOG_DEBUG("#########################################");
#endif
      
      while (lenv->running()) {
#ifdef DEBUG_WITH_DETERMINIST_POLICY
          if(! bib::Utils::equals(all_perceptions.front(),lenv->perceptions())){
            bib::Logger::PRINT_ELEMENTS(all_perceptions.front(), "allper");
            bib::Logger::PRINT_ELEMENTS(lenv->perceptions(), "curren pert");
          
            LOG_DEBUG("state discordance " << i << " " << this->agent->get_decision_each()*begin << " " << all_perceptions.size());
            all_perceptions.pop_front();
            
            bib::Logger::PRINT_ELEMENTS(all_perceptions.front(), "next allper");
          
            exit(1);
          }
          ASSERT(bib::Utils::equals(all_perceptions.front(),lenv->perceptions()), "state discordance " << i << " " << begin);
        
          LOG_DEBUG(all_perceptions.size());
          bib::Logger::PRINT_ELEMENTS(all_perceptions.front(), "DEBallper");
          bib::Logger::PRINT_ELEMENTS(lenv->perceptions(), "DEBcurren pert");
          bib::Logger::PRINT_ELEMENTS(*it, "DEBac");
        
          all_perceptions.pop_front();
#endif
          if(i >= begin * this->agent->get_decision_each()){
            std::vector<double> st = lenv->perceptions();
            std::vector<double>& ac = lpol->run_td(st);
//             bib::Logger::PRINT_ELEMENTS(ac, "DEBmac ");
//             LOG_DEBUG((lpol->did_decision() ? "true" : "false"));
            inter_rewards.push_back(lenv->performance());
            lenv->apply(ac);
            if(lpol->did_decision()){
              rsum += (local_gamma * (*std::max_element(inter_rewards.begin(), inter_rewards.end())));
              local_gamma *= gamma;
              inter_rewards.clear();
            }
          }
          else
            lenv->apply(*it);
          
          ++it;
          i++;
        }
        
        rsum += local_gamma * lenv->performance();
        mmean->at(mean) = rsum;
        
#ifdef DEBUG_WITH_DETERMINIST_POLICY
        if(begin == 0)
          ASSERT(fabs(swr - rsum) <= 1e-8, "sum_weighted_rewards discordance " << swr << " " << rsum << " " << (lenv->final_state() ? "true" : "false") <<" " <<local_gamma);
#endif
        
        lenv->next_instance_choose(stochasticity);
    }
    
    double s = 0;
    for(auto it : *mmean){
      s += it;
    }
    
    delete mmean;
    
    return s/precision;
  }
  
  
 private:
    bool analyse_distance_bestVF = false;
    uint precision = 0;
    uint shm_id;

};
}  // namespace arch

#endif