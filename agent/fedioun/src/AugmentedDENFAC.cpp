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
    if(trajectory.size() >= replay_memory)
      trajectory.pop_front();

    trajectory.push_back(sa);
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
    mini_batch_size             = pt->get<uint>("agent.mini_batch_size");
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
                  mini_batch_size,
                  decay_v,
                  hidden_layer_type, batch_norm,
                  weighting_strategy > 0);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, mini_batch_size, hidden_layer_type, last_layer_actor,
                  batch_norm);
}
  
/***
 * Load previous run to resume in case of interruption
 */
void AugmentedDENFAC::load_previous_run() {
	ann->load("continue.actor");
	qnn->load("continue.critic");
	auto p1 = bib::XMLEngine::load<std::deque<sample>>("trajectory", "continue.trajectory.data");
	trajectory = *p1;
	delete p1;

	auto p3 = bib::XMLEngine::load<struct algo_state>("algo_state", "continue.algo_state.data");
	mini_batch_size = p3->mini_batch_size;
	replay_memory = p3->replay_memory;
	delete p3;

	ann->increase_batchsize(mini_batch_size);
	qnn->increase_batchsize(mini_batch_size);

}

/***
 * Save current state 
 */
void AugmentedDENFAC::save_run() {
	ann->save("continue.actor");
	qnn->save("continue.critic");
	bib::XMLEngine::save(trajectory, "trajectory", "continue.trajectory.data");
	struct algo_state st = {mini_batch_size, replay_memory};
	bib::XMLEngine::save(st, "algo_state", "continue.algo_state.data");
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

	learning = _learning;

	step = 0;
}

/***
 * Compute importance sampling
 */
void AugmentedDENFAC::computePThetaBatch(const std::deque< sample >& vtraj, double *ptheta, const std::vector<double>* all_next_actions) {
	uint i=0;
	for(auto it : vtraj) {
	  double p0 = 1.f;
	  for(uint j=0; j < nb_motors; j++) {
	    p0 *= bib::Proba<double>::truncatedGaussianDensity(it.a[j], all_next_actions->at(i*nb_motors+j), noise);
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
	std::deque<sample>* traj = &trajectory;
	std::vector<double>* q_targets_weights = nullptr;

	std::vector<double> all_next_states(traj->size() * nb_sensors);
	std::vector<double> all_states(traj->size() * nb_sensors);
	std::vector<double>* 	all_actions = new std::vector<double>(traj->size() * nb_motors);
	double* ptheta = nullptr;

	std::vector<double>* q_targets;


	// ******* Compute Q targets *******

	if (!retrace_lambda) {

		double lambda = 0.8;
		std::vector<double>*			   all_next_actions;			// \pi(s_t)
		std::vector<double>*       					 all_QV;			// Q(s_t, \mu(s_t))
		std::vector<double>*                   next_action = new std::vector<double>(nb_motors);
		std::vector<double>* 			  randomized_action;
		std::vector<double>       all_next_QV(traj->size());			// Q(s_t, \pi(s_t))
		std::vector<double>     retrace_coefs(traj->size());				
		std::vector<double> bellman_residuals(traj->size());			
		q_targets = new std::vector<double>(traj->size()); // Retrace Q target

		
		uint i=0;

		// Get data from traj
		for (auto it : *traj) {
			std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + i * nb_sensors);
			std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
			std::copy(it.a.begin(), it.a.end(), all_actions->begin() + i * nb_motors);
			i++;
		}

		// Compute retrace_coefs (check other version later)
		all_next_actions = ann->computeOutBatch(all_next_states);

		// \pi(s_{t+1})
		ptheta = new double[traj->size()];
		computePThetaBatch(*traj, ptheta, all_next_actions);

		i = 0;
		for (auto it : *traj) {
			retrace_coefs[i] = lambda * std::min((double)1, ptheta[i] / it.p0 );
			i++;
		}

		// Compute Bellman residuals

		std::cout << "nb_motors : " << nb_motors << std::endl;
		std::cout << "nb_sensors : " << nb_sensors << std::endl;
		std::cout << "nb_states : " << traj->size() << std::endl;
		std::cout << "all_actions : " <<  all_actions->size() << std::endl;
		std::cout << "all_states : " << all_states.size() << std::endl;


		// Q (s_t,a_t)
		all_QV = qnn->computeOutVFBatch(all_states, *all_actions);

		// Q (s_{t+1}, a_{t+1})

		uint nb_Q_samples = 1;
		

		for (uint j = 0; j < traj->size(); j++) {
			// Copy next action
			for(uint k = 0; k < nb_motors; k++) {
				next_action->at(k) = all_next_actions->at(j*nb_motors +k);
			}
			// Add gaussian noise and compute corresponding Q_value
			double avg_QV = 0;
			// Sampling
			for (uint k = 0; k < nb_Q_samples; k++) {
				randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
				avg_QV += qnn->computeOutVF(traj->at(j).next_s, *randomized_action);
			}
			all_next_QV.at(j) = avg_QV / nb_Q_samples;
		}

		i = 0;
		// TODO compute action, (add noise + compute Q value) * k
		for (auto it = traj->begin(); it != traj->end(); it++) {
			bellman_residuals[i] = it->r;
			if(!it->goal_reached) {
				bellman_residuals[i] += gamma * all_next_QV.at(i) - all_QV->at(i); 
			}
			i++;		
		}

		// Compute Q targets

		// \delta Q(x_t,a_t) = \sum_{s=t}^{t+k-1} \gamma^{s-t} ( \prod_{i = t+1}^s c_i )[ r(x_s, a_s) + \gamma Q(x_{s+1}, a_{s+1}) - Q(x_s, a_s)]

		double retrace_target_sum = 0;
		// TODO goal reached
		for(int j = traj->size()-1; j >= 0; j--) {
			retrace_target_sum += bellman_residuals[j];
			q_targets->at(j) = retrace_target_sum;

			retrace_target_sum *= gamma * retrace_coefs[j];			
		}

		delete next_action;
		delete all_next_actions;
		delete all_QV;
		delete randomized_action;
		delete[] ptheta; 
	} else {

		// Standard DENFAC Q target

		// Compute \pi(s_{t+1})

		// Get batch data (s_t, s_{t+1} and a)
		uint i=0;
		for (auto it : *traj) {
			std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + i * nb_sensors);
			std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
			std::copy(it.a.begin(), it.a.end(), all_actions->begin() + i * nb_motors);
			i++;
		}

		// Compute all next action
		std::vector<double>* all_next_actions;
		all_next_actions = ann->computeOutBatch(all_next_states);

		// Compute next Q value

		q_targets = qnn->computeOutVFBatch(all_next_states, *all_next_actions);

		// Importance sampling
		if(weighting_strategy != 0) {
			q_targets_weights = new std::vector<double>(q_targets->size(), 1.0f);
			if(weighting_strategy > 1) {
				ptheta = new double[traj->size()];
				computePThetaBatch(*traj, ptheta, all_next_actions);
			}
		}
		delete all_next_actions;

		// Adjust q_targets
		i=0;
		for (auto it : *traj) {
			if(it.goal_reached)
				q_targets->at(i) = it.r;
			else {
				q_targets->at(i) = it.r + gamma * q_targets->at(i);
			}
			if(weighting_strategy==1)
				q_targets_weights->at(i)=1.0f/it.p0;
			else if(weighting_strategy==2)
				q_targets_weights->at(i)=ptheta[i]/it.p0;
			else if(weighting_strategy==3)
				q_targets_weights->at(i)=std::min((double)1.0f, ptheta[i]/it.p0);
			i++;
		}





	}

	// Optionnaly reset the network 
	if(reset_qnn && episode < 1000 ) {
		delete qnn;
		qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
		          alpha_v,
		          mini_batch_size,
		          decay_v,
		          hidden_layer_type, batch_norm,
		          weighting_strategy > 0);
  	}

  	if (retrace_lambda) {
  		const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
		double* q_values_diff = q_values_blob->mutable_cpu_diff();

		for (uint j =0; j < traj->size(); j++) {			
			q_values_diff[q_values_blob->offset(j,0,0,0)] = q_targets->at(j);
		}

		qnn->critic_backward();
		// Update QTM
		ann->getSolver()->ApplyUpdate();
		ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);

		qnn->ZeroGradParameters();
		std::cout << "Critic update done" << std::endl;
		delete q_targets;


  	} else {
  		// Update critic
		if(weighting_strategy != 0)
			qnn->stepCritic(all_states, *all_actions, *q_targets, iter, q_targets_weights);
		else
			qnn->stepCritic(all_states, *all_actions, *q_targets, iter);

		delete q_targets;
		if(weighting_strategy != 0) {
			delete q_targets_weights;
			if(weighting_strategy > 1)
		  		delete[] ptheta;
		}

		qnn->ZeroGradParameters();
  	}


  	delete all_actions;

}

/***
 * Compute the critic's gradient wrt the actor's actions
 * Update the actor using this gradient and the inverting gradient strategy 
 */
void AugmentedDENFAC::actor_update_grad() {

	std::deque<sample>* traj = &trajectory;

	std::vector<double> all_states(traj->size() * nb_sensors);
	uint i=0;
	for (auto it : *traj) {
		std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
		i++;
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
	i=0;
	for (auto it : *traj)
		q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;

	// Compute s and a toward an increase of Q
	qnn->critic_backward();

	// Get a
	const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);

	// Inverting gradient strategy
	if(inverting_grad) {
		// QTM
		double* action_diff = critic_action_blob->mutable_cpu_diff();

		for (uint n = 0; n < traj->size(); ++n) {
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

	if(trajectory.size() != mini_batch_size) {
	    mini_batch_size = trajectory.size();
	    qnn->increase_batchsize(mini_batch_size);
	    ann->increase_batchsize(mini_batch_size);
  	}

	// Fitted updates
	for(uint n=0; n<nb_fitted_updates; n++) {

		// Fitted critic updates
		for(uint i=0; i<nb_critic_updates ; i++)
			critic_update(nb_internal_critic_updates);

		if(reset_ann && episode < 1000) {
			delete ann;
			ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, mini_batch_size, hidden_layer_type, last_layer_actor, batch_norm);
		}

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
	    << " " << trajectory.size()
	    << " " << ann->weight_l1_norm()
	    << " " << std::fixed << std::setprecision(7) << qnn->error()
	    << " " << qnn->weight_l1_norm()
	#endif
	    ;
}

void AugmentedDENFAC::_dump(std::ostream& out) const  {
	out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
	sum_weighted_reward << " " << std::setw(8) << std::fixed <<
	std::setprecision(5) << trajectory.size() ;
}

double AugmentedDENFAC::criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions){
    return qnn->computeOutVF(perceptions, actions);
}

arch::Policy<MLP>* AugmentedDENFAC::getCopyCurrentPolicy() {
	return new arch::Policy<MLP>(new MLP(*ann, true) , gaussian_policy ? arch::policy_type::GAUSSIAN :
                             arch::policy_type::GREEDY,
                             noise, decision_each);
}