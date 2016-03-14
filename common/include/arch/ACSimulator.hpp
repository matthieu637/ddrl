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

//#define NOPARA

using std::string;

namespace arch {

template <typename Environment, typename Agent, typename Stat=DummyEpisodeStat>
class ACSimulator : public Simulator<Environment, Agent, Stat> {

  typedef Policy< typename Agent::PolicyImpl> _Policy;

 public:
  ACSimulator() : arch::Simulator<Environment, Agent, Stat>() {
        LOG_INFO("actor-critic simulator should only be used for policy/value function analyse");
        LOG_INFO("hypothesis : policy doesn't change online, determinist environment, only one instance");
  }

  ~ACSimulator() {
#ifdef NOPARA
    if(analyse_distance_bestVF){
      for(auto it : addi_env){
        delete it;
      }
    }
#endif
  }

  void enable_analyse_distance_bestVF(){
      if(this->properties == nullptr){
          LOG_ERROR("call enable_analyse_distance_bestVF after init");
          exit(1);
      }
    
      analyse_distance_bestVF = true;
      
#ifdef NOPARA
      while(addi_env.size() < 1)
      {
        Environment *lenv = new Environment;
        lenv->unique_invoke(this->properties, this->command_args);
        addi_env.push_back(lenv);
      }
#endif
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
        double diff = compareBestValueFonction(all_perceptions, all_actions, policy, this->agent->getGamma(), all_Vs);
        //double diff = 0;
      
        LOG_DEBUG("diff (higher bad) " << diff);
        LOG_FILE("ac.data", std::setprecision(12) << diff);
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
    
#ifdef NOPARA
  struct ParraBestVFunction{
    ParraBestVFunction( _Policy* policy, std::list< std::vector<double> >& _all_actions, double _gamma, ACSimulator<Environment, Agent, Stat>* _ptr, uint para):
        pols(1), all_actions(_all_actions), gamma(_gamma), ptr(_ptr){
      //pre build policies to avoid over copying
      for(uint i=0;i<1;i++)
        pols[i] =  new _Policy(*policy);
      
      mmean = new vector<double>(para);
    }
    
    ~ParraBestVFunction() { //must be empty cause of tbb

    }
    
    void operator()(const tbb::blocked_range<int>& range) const {
      for (int i = range.begin(); i < range.end(); i++){
        mmean->at(i) = ptr->evalBestValueFonction(all_actions, pols, gamma, i);
      }
      
    }
    
    void free() {
      for(uint i=0;i<1;i++)
        delete pols[i];
      
      delete mmean;
    }
    
    vector<_Policy*> pols;
    std::list< std::vector<double> >& all_actions;
    double gamma;
    ACSimulator<Environment, Agent, Stat>* ptr;
    std::vector<double>* mmean;
  };
#endif
  
  double compareBestValueFonction(std::list< std::vector<double> > all_perceptions, 
                                          std::list< std::vector<double> > all_actions, _Policy* policy, double gamma,
                                          std::list< double > all_Vs
                                 ){
    
    double diff = 0;
#ifdef NOPARA
    ParraBestVFunction pbvf(policy, all_actions, gamma, this, all_Vs.size());
    
    //tbb::parallel_for(tbb::blocked_range<int>(0, all_Vs.size(), all_Vs.size()/bib::ThreadTBB::getInstance()->get_number_thread()), pbvf);
    pbvf(tbb::blocked_range<int>(0, all_Vs.size()));
    
    int i=0;
    for(auto it_vs : all_Vs){
      //double v = evalBestValueFonction(all_actions, pols, gamma, i);
      //diff += (v-it_vs)*(v-it_vs);
      diff += (pbvf.mmean->at(i)-it_vs)*(pbvf.mmean->at(i)-it_vs);
      i++;
      ++it_vs;
    }
    
    pbvf.free();
#else
    
    using namespace boost::interprocess;
    typedef allocator<double, managed_shared_memory::segment_manager>  ShmemAllocator;
    typedef boost::interprocess::vector<double, ShmemAllocator> MyVector;
    
    shared_memory_object::remove("ac_sim_forked");
    
    managed_shared_memory shm_obj(create_only, "ac_sim_forked", 65536 );
    const ShmemAllocator alloc_inst (shm_obj.get_segment_manager());
    
    MyVector *myvector = shm_obj.construct<MyVector>("result")(alloc_inst);
    
    for(uint i = 0; i < all_Vs.size(); ++i)
      myvector->push_back(0);
    
    //double *gamma_ = shm_obj.construct<double>("gamma")(alloc_inst);
    //*gamma_ = gamma;
    double* _gamma = new double;
    *_gamma = gamma;
   
    std::vector<int> childs_pid;
    
    uint work_per_process = all_Vs.size() / bib::ThreadTBB::getInstance()->get_number_thread();
    uint additional_work = all_Vs.size() - (work_per_process * bib::ThreadTBB::getInstance()->get_number_thread());
    
    for (uint i = 0; i < bib::ThreadTBB::getInstance()->get_number_thread(); i++){
      int pid = fork();
      if(pid ==0){
        //Open the managed segment
        managed_shared_memory segment2(open_only, "ac_sim_forked");

        //Find the vector using the c-string name
        MyVector *myvector2 = segment2.find<MyVector>("result").first;

        //Use vector in reverse order
        myvector2->at(0) = 10;
        
        Environment *lenv = new Environment;
        lenv->unique_invoke(this->properties, this->command_args);
        addi_env.push_back(lenv);
        std::vector<_Policy*> pols(1);
        pols[0] =  new _Policy(*policy);
        for(uint j=0;j < work_per_process; j++){
          uint z = (i*work_per_process) + j;
          myvector2->at(z) = evalBestValueFonction(all_actions, pols, gamma, z);
        }
        
        if(i == 0){
          for(uint j=0;j < additional_work; j++){
            uint z = (bib::ThreadTBB::getInstance()->get_number_thread()*work_per_process) + j;
            myvector2->at(z) = evalBestValueFonction(all_actions, pols, gamma, z);
          }
        }
        
        exit(0);
      } else {
       childs_pid.push_back(pid);
      }
    }
    
    for(auto it : childs_pid)
      waitpid(it, nullptr, 0);

    int i=0;
    for(auto it_vs : all_Vs){
      //double v = evalBestValueFonction(all_actions, pols, gamma, i);
      //diff += (v-it_vs)*(v-it_vs);
      diff += (myvector->at(i)-it_vs)*(myvector->at(i)-it_vs);
      i++;
      ++it_vs;
    }
    
    //LOG_DEBUG(myvector->at(0));
    
    shared_memory_object::remove("ac_sim_forked");
#endif
    
    return diff / all_perceptions.size();
  }

  struct ParraBestV {
  ParraBestV(std::list< std::vector<double> >& _all_actions, const vector<_Policy*>& _policy, 
              double _gamma, int _begin, ACSimulator<Environment, Agent, Stat>* _ptr) : 
          all_actions(_all_actions), policy(_policy), gamma(_gamma), begin(_begin), ptr(_ptr)  {
    mmean = new std::vector<double>(ptr->precision);
  }

  ~ParraBestV() { //must be empty cause of tbb

  }

  void free() {
    delete mmean;
  }

  void operator()(const tbb::blocked_range<int>& range) const {
    Environment* lenv = ptr->addi_env[bib::ThreadTBB::getInstance()->get_my_thread_id()];
    _Policy* lpol = policy[bib::ThreadTBB::getInstance()->get_my_thread_id()];
    
    
    for (int mean = range.begin(); mean < range.end(); mean++) {
      lenv->reset_episode();
      
      auto it = all_actions.cbegin();
      double local_gamma = 1.00000000000000f;
      double rsum = 0.00000000000000f;
      int i = 0;
      while (lenv->running()) {
        if(i >= begin){
          std::vector<double> st = lenv->perceptions();
          std::vector<double>* ac = lpol->run(st);
          lenv->apply(*ac);
          delete ac;
          rsum += (local_gamma * lenv->performance());
        }
        else
          lenv->apply(*it);
        
          local_gamma *= gamma;
          ++it;
          i++;
        }
        mmean->at(mean) = rsum;
        
        lenv->next_instance();
      }
      //LOG_DEBUG(bib::ThreadTBB::getInstance()->get_my_thread_id());
      //LOG_DEBUG(bib::ThreadTBB::getInstance()->get_my_thread_id() << "  work from  " << range.begin() << " to " << range.end() << " " << (range.end()-range.begin())  );
    }

    std::list< std::vector<double> >& all_actions;
    const vector<_Policy*>& policy;
    double gamma;
    int begin;
    
    ACSimulator<Environment, Agent, Stat>* ptr;
    std::vector<double>* mmean;
  };
  
  
  double evalBestValueFonction(std::list< std::vector<double> > all_actions, 
                               const vector<_Policy*>& policy, double gamma, int begin){
    ParraBestV dq(all_actions, policy, gamma, begin, this);
    
    //don't parallel at this level it's too quick
    //tbb::parallel_for(tbb::blocked_range<int>(0, precision, precision/bib::ThreadTBB::getInstance()->get_number_thread()), dq, tbb::simple_partitioner());
    dq(tbb::blocked_range<int>(0, precision));
    
    double s = 0;
    for(auto it : *dq.mmean){
      s+= it;
    }
    
    dq.free();
    
    return s/precision;
  }
  
  
 private:
    bool analyse_distance_bestVF = false;
    int precision = 100;
    std::vector<Environment*> addi_env;

};
}  // namespace arch

#endif
