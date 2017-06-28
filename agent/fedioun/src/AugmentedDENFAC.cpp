#include "AugmentedDENFAC.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>

#include "nn/MLP.hpp"
#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "bib/IniParser.hpp"

#define DOUBLE_COMPARE_PRECISION 1e-9

typedef MLP PolicyImpl;

AugmentedDENFAC::AugmentedDENFAC(unsigned int _nb_motors, unsigned int _nb_sensors): arch::AACAgent<PolicyImpl, AgentGPUProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {
}

AugmentedDENFAC::~AugmentedDENFAC() {
	delete qnn;
	delete ann;

	delete hidden_unit_q;
	delete hidden_unit_a;
}


/***
 *	Compute next action with an exploration strategy
 *  Update last action and last state and best reward
 *  Fill replay buffer
 */
const std::vector<double>& AugmentedDENFAC::_run(double reward, const std::vector<double>& sensors, bool learning, bool goal_reached, bool last) {
    
    // Store previous sample into the replay buffer 
    if (last_action.get() != nullptr && learning) {
    	double p0 = 1.f;
	    for(uint i=0; i < nb_motors; i++) {
	        p0 *= bib::Proba<double>::truncatedGaussianDensity(last_action->at(i), last_pure_action->at(i), noise);
	    }
	    // TODO check reward terminal state
        sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0};
        insertSample(sa);
    }

    // Compute next action
    vector<double>* next_action = ann->computeOut(sensors);
    // Update last action
    last_pure_action.reset(new vector<double>(*next_action));

    // Add exploration strategy
    if(learning) {    	
      if(gaussian_policy) {
      	// Gaussian noise
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(bib::Utils::rand01() < noise) { 
      	//e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }

    // Update last action
    last_action.reset(next_action);	

    // Update last state
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);
    return *next_action;
}

/***
 * Insert a sample into the replay buffer
 * Save trajectories
 * Called from _run()
 * Optionnaly do on-line updates
 */
void AugmentedDENFAC::insertSample(const sample& sa) {
	//LOG_DEBUG(trajectories.size());
	//LOG_DEBUG(trajectories.back().transitions->size());
	((trajectories.back()).transitions)->push_back(sa);

	/*
    if(trajectories.size() >= replay_memory)
      trajectory.pop_front();

    trajectory.push_back(sa);
    */
}


/***
 * Get parameters from config.ini (located in the current build directory or provided with the --config option)
 * Initialize actor and critic networks
 */
void AugmentedDENFAC::_unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) {
    hidden_unit_q               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                       = pt->get<double>("agent.noise");
    gaussian_policy             = pt->get<bool>("agent.gaussian_policy");
    replay_memory               = pt->get<uint>("agent.replay_memory");
    reset_qnn                   = pt->get<bool>("agent.reset_qnn");
    nb_actor_updates            = pt->get<uint>("agent.nb_actor_updates");
    nb_critic_updates           = pt->get<uint>("agent.nb_critic_updates");
    nb_fitted_updates           = pt->get<uint>("agent.nb_fitted_updates");
    nb_internal_critic_updates  = pt->get<uint>("agent.nb_internal_critic_updates");
    alpha_a                     = pt->get<double>("agent.alpha_a");
    alpha_v                     = pt->get<double>("agent.alpha_v");
    batch_norm                  = pt->get<uint>("agent.batch_norm");
    inverting_grad              = pt->get<bool>("agent.inverting_grad");
    decay_v                     = pt->get<double>("agent.decay_v");
    weighting_strategy          = pt->get<uint>("agent.weighting_strategy");
    last_layer_actor            = pt->get<uint>("agent.last_layer_actor");
    reset_ann                   = pt->get<bool>("agent.reset_ann");
    hidden_layer_type           = pt->get<uint>("agent.hidden_layer_type");
    replay_traj_size			= pt->get<uint>("agent.replay_traj_size");

    retrace_lambda				= pt->get<bool>("agent.retrace_lambda");
    
	#ifdef CAFFE_CPU_ONLY
	    LOG_INFO("CPU mode");
	    (void) command_args;
	#else
	    if(command_args->count("gpu") == 0 || command_args->count("cpu") > 0) {
	      caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
	      LOG_INFO("CPU mode");
	    } else {
	      caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
	      caffe::Caffe::SetDevice(0);
	      LOG_INFO("GPU mode");
	    }
	#endif

    qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
                  alpha_v,
                  1,
                  decay_v,
                  hidden_layer_type, batch_norm,
                  weighting_strategy > 0);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, 1, hidden_layer_type, last_layer_actor,
                  batch_norm);
}
  
/***
 * Load previous run to resume in case of interruption
 * TO DO : Trajectories data struct
 */
void AugmentedDENFAC::load_previous_run() {
	ann->load("continue.actor");
	qnn->load("continue.critic");
	//auto p1 = bib::XMLEngine::load<std::deque<sample>>("trajectory", "continue.trajectory.data");
	//trajectory = *p1;
	//delete p1;

	auto p3 = bib::XMLEngine::load<struct algo_state>("algo_state", "continue.algo_state.data");
	//replay_memory = p3->replay_memory;
	delete p3;

}

/***
 * Save current state 
 * TO DO : Trajectories data struct
 */
void AugmentedDENFAC::save_run() {
	ann->save("continue.actor");
	qnn->save("continue.critic");
	//bib::XMLEngine::save(trajectory, "trajectory", "continue.trajectory.data");
}

/***
 * Episode initialization
 */
void AugmentedDENFAC::_start_episode(const std::vector<double>& sensors, bool _learning) {
	last_state.clear();
	for (uint i = 0; i < sensors.size(); i++)
	  last_state.push_back(sensors[i]);

	last_action = nullptr;
	last_pure_action = nullptr;

	// Updating the replay buffer
	if(trajectories.size() >= replay_traj_size) {
		trajectories.pop_front();		
	}

	if (_learning) {
		trajectory traj;
		traj.transitions.reset(new std::vector<sample>);

		trajectories.push_back(traj);
		//LOG_DEBUG(trajectories.size());
	}
	

	learning = _learning;

	step = 0;
}

/***
 * Compute importance sampling
 */
void AugmentedDENFAC::computePThetaBatch(const std::vector< sample >& vtraj, double *ptheta, const std::vector<double>* all_next_actions) {
	uint i=0;
	for(auto it : vtraj) {
	  double p0 = 1.f;
	  for(uint j=0; j < nb_motors; j++) {
	  	//std::cout << "At " << j << " : p0 = " << p0 << " PastAction = " << it.a[j] << " NextAction = " << all_next_actions->at(i*nb_motors+j) << std::endl;
	    p0 *= bib::Proba<double>::truncatedGaussianDensity(it.a[j], all_next_actions->at(i*nb_motors+j), noise);
	    	//std::cout << "TruncatedDensityFunction = " << bib::Proba<double>::truncatedGaussianDensity(it.a[j], all_next_actions->at(i*nb_motors+j), noise) << std::endl;
	    	}

	  ptheta[i] = p0;


	  i++;

	}
}

/***
 * Update the critic using a trajectory sampled from the replay buffer
 */
void AugmentedDENFAC::critic_update(uint iter) {
	// Get Replay buffer
	std::deque<trajectory>* traj = &trajectories;
	int traj_size = 0;


	std::vector<double>* 	trajectory_next_states;
	std::vector<double>* 	trajectory_states;
	std::vector<double>* 	trajectory_actions;
	std::vector<double>*	trajectory_next_actions;			// \pi(s_t)
	std::vector<double>* 	trajectory_q_targets;
	std::vector<double>*    retrace_coefs;
	std::vector<double>*    trajectory_QV;			// Q(s_t, \mu(s_t))
	std::vector<double>*    trajectory_next_QV;			// Q(s_{t+1}, \mu(s_{t+1}))
	std::vector<double>*    next_action;
	std::vector<double>* 	randomized_action;
	double* 				bellman_residuals;

	std::vector<double>     all_states;
	std::vector<double>     all_actions;
	std::vector<double>     all_q_targets;
	


	double* ptheta = nullptr;
	double lambda = 0.9;


	/*
	

	
	std::vector<double>       all_next_QV(traj_size);			// Q(s_t, \pi(s_t))
					
	std::vector<double> bellman_residuals(traj_size);

	q_targets = new std::vector<double>(traj_size); // Retrace Q target
	*/


	// For each trajectory
	for(auto current_traj : *traj) {
		// Get size of trajectories	
		traj_size += (current_traj.transitions)->size();
		//std::cout << "Traj " << current_traj.id << " size : " << (current_traj.transitions)->size() << std::endl;

		// Allocate memory
		trajectory_states = 		new std::vector<double>((current_traj.transitions)->size() * nb_sensors);
		trajectory_next_states = 	new std::vector<double>((current_traj.transitions)->size() * nb_sensors);
		trajectory_actions = 		new std::vector<double>((current_traj.transitions)->size() * nb_motors);
		retrace_coefs = 			new std::vector<double>((current_traj.transitions)->size());
		trajectory_next_QV =		new std::vector<double>((current_traj.transitions)->size());
		trajectory_q_targets = 		new std::vector<double>((current_traj.transitions)->size());
		next_action = 				new std::vector<double>(nb_motors);
		ptheta = 					new double[(current_traj.transitions)->size()];
		bellman_residuals = 		new double[(current_traj.transitions)->size()];

		// ******* Compute Q targets *******

		// Get data from traj
		uint i=0;
		for (auto it : *(current_traj.transitions)) {
			std::copy(it.next_s.begin(), it.next_s.end(), trajectory_next_states->begin() + i * nb_sensors);
			std::copy(it.s.begin(), it.s.end(), trajectory_states->begin() + i * nb_sensors);
			std::copy(it.a.begin(), it.a.end(), trajectory_actions->begin() + i * nb_motors);
			i++;
		}

		// Compute retrace_coefs (check other version later)
		ann->increase_batchsize((current_traj.transitions)->size());
		trajectory_next_actions = ann->computeOutBatch(*trajectory_next_states);

		// \pi(s_{t+1})
		
		computePThetaBatch(*(current_traj.transitions), ptheta, trajectory_next_actions);

		i = 0;
		for (auto it : *(current_traj.transitions)) {
			retrace_coefs->at(i) = lambda * std::min((double)1, ptheta[i] / it.p0 );
			//std::cout << "At " << i << " : ptheta = " << ptheta[i] << "  p0 = " << it.p0 << " r= " << ptheta[i] / it.p0 <<  std::endl;
			i++;
		}

		// Compute Bellman residuals :

		// Q (s_t,a_t)
		qnn->increase_batchsize((current_traj.transitions)->size());
		trajectory_QV = qnn->computeOutVFBatch(*trajectory_states, *trajectory_actions);

		// Q (s_{t+1}, a_{t+1})

		uint nb_Q_samples = 1;
		

		for (uint j = 0; j < (current_traj.transitions)->size(); j++) {
			// Copy next action
			for(uint k = 0; k < nb_motors; k++) {
				next_action->at(k) = trajectory_next_actions->at(j*nb_motors +k);
			}
			// Add gaussian noise and compute corresponding Q_value
			double avg_QV = 0;
			// Sampling
			for (uint k = 0; k < nb_Q_samples; k++) {
				randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
				
				avg_QV += qnn->computeOutVF((current_traj.transitions)->at(j).next_s, *randomized_action);
			}
			trajectory_next_QV->at(j) = avg_QV / nb_Q_samples;
		}

		i = 0;
		// TODO compute action, (add noise + compute Q value) * k
		for (auto it = (current_traj.transitions)->begin(); it != (current_traj.transitions)->end(); it++) {

			bellman_residuals[i] = it->r;
			//std::cout << "Reward : " << it->r;
			if(!it->goal_reached) {
				//std::cout << " Current QV : " << all_QV->at(i) << " Next QV : " << all_next_QV.at(i) << std::endl ;
				bellman_residuals[i] += gamma * trajectory_next_QV->at(i) - trajectory_QV->at(i); 
			}
			//std::cout << "Bellman residual : " << bellman_residuals[i] << std::endl;
			i++;
		}

		// Compute Q targets

		// \delta Q(x_t,a_t) = \sum_{s=t}^{t+k-1} \gamma^{s-t} ( \prod_{i = t+1}^s c_i )[ r(x_s, a_s) + \gamma Q(x_{s+1}, a_{s+1}) - Q(x_s, a_s)]

		double retrace_target_sum = 0;
		// TODO goal reached

		for(int j = (current_traj.transitions)->size()-1; j >= 0; j--) {
			// Q-targets computed separately for each trajectory

			retrace_target_sum += bellman_residuals[j];
			//std::cout << "Bellman Residual : " << bellman_residuals[j] << std::endl;
			
			trajectory_q_targets->at(j) = retrace_target_sum;
			//std::cout << "Q_target at " << j << " = " << retrace_target_sum << std::endl;
			retrace_target_sum *= gamma * retrace_coefs->at(j);
			//std::cout << "Retrace_coefs :  " << retrace_coefs[j] << std::endl << std::endl;		
		}

		// Saving data
		for (uint j = 0; j < (current_traj.transitions)->size(); j++) {
			all_q_targets.push_back(trajectory_q_targets->at(j));
			for (int i=0; i < nb_sensors; i++) {
				all_states.push_back(trajectory_states->at(j * nb_sensors + i));
			}
			for (int i=0; i < nb_motors; i++) {
				all_actions.push_back(trajectory_actions->at(j * nb_motors + i));
			}
		}

			// Free memory
		delete trajectory_states;
		delete trajectory_next_states;
		delete trajectory_actions;
		delete retrace_coefs;
		delete trajectory_next_QV;
		delete next_action;
		delete[] ptheta;
		delete[] bellman_residuals;


	}


	// Optionnaly reset the network 
	if(reset_qnn && episode < 1000 ) {
		delete qnn;
		qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
		          alpha_v,
		          1,
		          decay_v,
		          hidden_layer_type, batch_norm,
		          weighting_strategy > 0);
  	}

  	qnn->increase_batchsize(traj_size);
  	qnn->stepCritic(all_states, all_actions, all_q_targets, iter);

  	/*
	const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
	double* q_values_diff = q_values_blob->mutable_cpu_diff();

	for (uint j =0; j < traj->size(); j++) {			
		q_values_diff[q_values_blob->offset(j,0,0,0)] = q_targets->at(j);
	}

	qnn->critic_backward();

	// Update QTM
	ann->getSolver()->ApplyUpdate();
	ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);

	*/

	qnn->ZeroGradParameters();
	//std::cout << "Critic update done" << std::endl;



}

/***
 * Compute the critic's gradient wrt the actor's actions
 * Update the actor using this gradient and the inverting gradient strategy 
 */
void AugmentedDENFAC::actor_update_grad() {
	std::vector<double> all_states;

	int traj_size = 0;
	for(auto current_traj : trajectories) {
		traj_size += (current_traj.transitions)->size();
		for (auto it : *(current_traj.transitions)) {
			for (int i = 0; i < nb_sensors; i++) {
				all_states.push_back((it.s).at(i)); 
			}
		}
		// Get size of trajectories	

	}

	//Update actor
	qnn->ZeroGradParameters();
	ann->ZeroGradParameters();

	// Compute a
	auto all_actions_outputs = ann->computeOutBatch(all_states);

	// Compute q
	delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);

	const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
	double* q_values_diff = q_values_blob->mutable_cpu_diff();

	// Compute \nabla_a Q(s_t,a)
	for (int i = 0; i < traj_size ; i++) {
			q_values_diff[q_values_blob->offset(i,0,0,0)] = -1.0f;
	}

	// Compute s and a toward an increase of Q
	qnn->critic_backward();

	// Get a
	const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);

	// Inverting gradient strategy
	if(inverting_grad) {
		// QTM
		double* action_diff = critic_action_blob->mutable_cpu_diff();

		for (int n = 0; n < traj_size ; n++) {
			for (uint h = 0; h < nb_motors; ++h) {
				int offset = critic_action_blob->offset(n,0,h,0);
				double diff = action_diff[offset];
				double output = all_actions_outputs->at(offset);
				double min = -1.0;
				double max = 1.0;
				if (diff < 0) {
					diff *= (max - output) / (max - min);
				} else if (diff > 0) {
					diff *= (output - min) / (max - min);
				}
				action_diff[offset] = diff;
			}
		}
	}

	// Transfer input-level diffs from Critic to Actor

	// Get actor's output
	const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
	// Set the actor's output to the action value computed by the critic through the backward pass
	actor_actions_blob->ShareDiff(*critic_action_blob);
	// Propagate the difference
	ann->actor_backward();

	// Update QTM
	ann->getSolver()->ApplyUpdate();
	ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);

	delete all_actions_outputs;
}

/***
 * Performs fitted updates of the actor and the critic
 */
void AugmentedDENFAC::update_actor_critic() {
	if(!learning)
		return;

	int traj_size = 0;
	for(auto current_traj : trajectories)
		traj_size += (current_traj.transitions)->size();
	
	qnn->increase_batchsize(traj_size);
  	

	// Fitted updates
	for(uint n=0; n<nb_fitted_updates; n++) {

		// Fitted critic updates
		for(uint i=0; i<nb_critic_updates ; i++)
			critic_update(nb_internal_critic_updates);

		if(reset_ann && episode < 1000) {
			delete ann;
			ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, 1, hidden_layer_type, last_layer_actor, batch_norm);
		}

		ann->increase_batchsize(traj_size);
		// Fitted actor updates
		for(uint i=0; i<nb_actor_updates ; i++)
			actor_update_grad();
	}
}

/***
 * Start update
 */
void AugmentedDENFAC::end_episode(bool)  {
	episode++;
	update_actor_critic();
}

void AugmentedDENFAC::save(const std::string& path, bool)  {
	ann->save(path+".actor");
	qnn->save(path+".critic");
	LOG_INFO("Saved as " + path+ ".actor");
	//      bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
}

void AugmentedDENFAC::load(const std::string& path)  {
	ann->load(path+".actor");
	qnn->load(path+".critic");
}

void AugmentedDENFAC::_display(std::ostream& out) const  {
	out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward
	#ifndef NDEBUG
	    << " " << std::setw(8) << std::fixed << std::setprecision(5) << noise
	   // << " " << trajectory.size()
	    << " " << ann->weight_l1_norm()
	    << " " << std::fixed << std::setprecision(7) << qnn->error()
	    << " " << qnn->weight_l1_norm()
	#endif
	    ;
}

void AugmentedDENFAC::_dump(std::ostream& out) const  {
	/*
	out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
	sum_weighted_reward << " " << std::setw(8) << std::fixed << 
	std::setprecision(5) << trajectory.size() ;
	*/
}

double AugmentedDENFAC::criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions){
    return qnn->computeOutVF(perceptions, actions);
}

arch::Policy<MLP>* AugmentedDENFAC::getCopyCurrentPolicy() {
	return new arch::Policy<MLP>(new MLP(*ann, true) , gaussian_policy ? arch::policy_type::GAUSSIAN :
                             arch::policy_type::GREEDY,
                             noise, decision_each);
}