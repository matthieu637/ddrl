
// DEAD CODE TO REMOVE

void shrink_actions(vector<double>* next_action) {
  for(uint i=0; i < nb_motors ; i++)
    if(next_action->at(i) > 1.f)
      next_action->at(i)=1.f;
    else if(next_action->at(i) < -1.f)
      next_action->at(i)=-1.f;
}

void write_policy_file(uint index) {
  std::string file = "debug_policy." + std::to_string(index);
  std::ofstream out;
  out.open(file, std::ofstream::out);

  auto iter = [&](const std::vector<double>& x) {
    for(uint i=0; i < x.size(); i++)
      out << x[i] << " ";

    //vector<double> * ac = ann->computeOut(x);
    vector<double> * ac = policy(x);
    out << ac->at(0);
    out << std::endl;
    delete ac;
  };

  bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 100);
  out.close();
}

void update_actor_discretization(uint) {
  struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);

  uint n=0;
  for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
    sample sm = *it;

    for (uint i = 0; i < nb_sensors ; i++)
      data->input[n][i] = sm.s[i];

    std::vector<double>* na = vnn->optimizedBruteForce(sm.s,0.01);

    for (uint i = 0; i < nb_motors; i++)
      data->output[n][i] = na->at(i);

    //         LOG_FILE("debug_discret." + std::to_string(episode)+ "."+std::to_string(minep), sm.s[0] << " "<< na->at(0));

    delete na;

    n++;
  }

  ann->learn_stoch(data, 10000, 0, 0.00000001);

  fann_destroy_train(data);
}

bool is_feasible(const double* parameters, uint N) {
  for(uint i=0; i < N; i++)
    if(fabs(parameters[i]) >= 5.f) {
      return false;
    }

  return true;
}

double fitfun_sum_overtraj() {
  double sum = 0;
  for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
    sample sm = *it;

    //vector<double>* next_action = ann->computeOut(sm.s);
    vector<double>* next_action = policy(sm.s);

    sum += vnn->computeOutVF(sm.s, *next_action);
    //       sum += vnn->computeOutVF(sm.s, *next_action) * (1.f/proba_s.pdf(sm.s));

    delete next_action;
  }

  return sum / trajectory.size();
}

double fitfun(double const *x, int N) {
  const uint dimension = N;

  struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()),
                                        sizeof(struct fann_connection));
  fann_get_connection_array(ann->getNeuralNet(), connections);

  ASSERT(dimension == fann_get_total_connections(ann->getNeuralNet()), "dimension mismatch");
  for(uint j=0; j< dimension; j++)
    connections[j].weight=x[j];

  fann_set_weight_array(ann->getNeuralNet(), connections, dimension);
  free(connections);

  double sum = fitfun_sum_overtraj();

  //LOG_DEBUG(sum);
  return -sum;
}

void update_actor_cmaes() {
  cmaes_t evo;
  double *arFunvals, *const*pop;
  const double* xfinal;
  uint i;

  struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()),
                                        sizeof(struct fann_connection));
  fann_get_connection_array(ann->getNeuralNet(), connections);
  const uint dimension = (nb_sensors+1)*hidden_unit_a + (hidden_unit_a+1)*nb_motors;
  double* startx  = new double[dimension];
  double* deviation  = new double[dimension];
  for(uint j=0; j< dimension; j++) {
    startx[j] = connections[j].weight;
    deviation[j] = 0.3;
  }
  free(connections);

  uint population = 15;
  uint generation = 50;
  arFunvals = cmaes_init(&evo, dimension, startx, deviation, 0, population, NULL);

  //run cmaes
  while(!cmaes_TestForTermination(&evo) && generation > 0 ) {
    pop = cmaes_SamplePopulation(&evo);
    generation--;

    for (i = 0; i < population; ++i)
      //           while (!is_feasible(pop[i], dimension)){
      cmaes_ReSampleSingle(&evo, i);
    //         }

    for (i = 0; i < cmaes_Get(&evo, "lambda"); ++i) {
      arFunvals[i] = fitfun(pop[i], dimension);
    }

    cmaes_UpdateDistribution(&evo, arFunvals);

    if(cmaes_TestForTermination(&evo))
      LOG_INFO("mismatch "<< cmaes_TestForTermination(&evo));
  }

  //get final solution
  xfinal = cmaes_GetPtr(&evo, "xbestever");

  connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()),
                sizeof(struct fann_connection));
  fann_get_connection_array(ann->getNeuralNet(), connections);

  ASSERT(dimension == fann_get_total_connections(ann->getNeuralNet()), "dimension mismatch");
  for(uint j=0; j< dimension; j++)
    connections[j].weight=xfinal[j];

  fann_set_weight_array(ann->getNeuralNet(), connections, dimension);
  free(connections);
}

void sample_transition(std::vector<sample>& traj) {
  for(uint i=0; i<traj.size(); i++) {
    int r = std::uniform_int_distribution<int>(0, trajectory.size() - 1)(*bib::Seed::random_engine());
    traj[i] = trajectory[r];
  }
}

//  Eval perf:
//  close all; clear all; X=load('../0.learning.data');plot(X(:, 4)); figure; X=save_best_policy(X(:,4)'); plot(X);
//  Debug pol
// close all; clear all; X=load_dirs('../', 'debug_policy.*', 2,0,0); index=-1:(1/((100-1)/(1-(-1)))):1; for i=1:size(X,1) figure; plot(index,X(i,:)); endfor
// close all; clear all; X=load_dirs('../', 'debug_policy.*', 2,0,0); index=-1:(1/((100-1)/(1-(-1)))):1; for i=1:size(X,1) figure; plot(index,X(i,:)); axis([-1 1 -1.5 1.5]); endfor
uint episode=0;
struct my_pol {
  MLP* ann;
  double J;
  uint index;

  bool operator< (const my_pol& b) const {
    return J < b.J;
  }
};
std::set<my_pol> old_policies;

uint andme= 0;

void end_episode() override {
#warning resample policies
  old_policies.insert({new MLP(*ann), sum_weighted_reward, episode});

  //     ann = new MLP(*old_policies.rbegin()->ann);
  //     LOG_DEBUG("SELECT policy "<< old_policies.rbegin()->index << " with score " << old_policies.rbegin()->J);

  if(!learning)
    return;

  episode++;
  //     write_policy_file(andme++);

  std::vector<MLP*> proposed_policies;
  std::vector<double> scores_sum_q;

  int* resetlooping;
  auto iter = [&]() {
    update_critic();
    //         d1= fitfun_sum_overtraj();
    double d1= fitfun_sum_overtraj();
    //         double d1= 0;
    LOG_DEBUG(ann->weight_l1_norm() << " " << d1 << " " << fann_get_MSE(vnn->getNeuralNet()) <<" "  <<
              vnn->weight_l1_norm());
    scores_sum_q.push_back(d1);
    //         proposed_policies.push_back(new MLP(*ann));

    update_actor_nfqca();

    //         update_actor_discretization(0);
    //         if(episode>20)
    //         write_policy_file(andme++);

    //         double d2 = fitfun_sum_overtraj();

    //         LOG_DEBUG("before " << d1 << " after " << d2 << (d2 >= d1 ? " IMPROVE" : " DEPROVE !!!"));
    //           update_actor_cmaes();
    //           update_actor_discretization(*resetlooping);
    //           update_actorN2();
    //         LOG_FILE("test" + std::to_string(episode) , d1 << " " << fann_get_MSE(vnn->getNeuralNet()));

    //         if(ann->weight_l1_norm() > 250.f && false){
    //           fann_randomize_weights(ann->getNeuralNet(), -0.025, 0.025);
    //           fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
    //           LOG_DEBUG("actor failed");
    //           *resetlooping = 0;
    //         }
  };

  //       auto eval = [&]() {
  //
  //
  //         double sr = fitfun_sum_overtraj();
  // //         if(fann_get_MSE(vnn->getNeuralNet()) < 0.001)
  //           return -sr;
  // //         else
  // //           return -sr+100. ;
  // //         return fann_get_MSE(vnn->getNeuralNet());
  //       };
  //
  //       NN best_ann = nullptr;
  //       NN best_vnn = nullptr;
  //       auto save_best = [&]() {
  //         if(best_ann != nullptr){
  //           fann_destroy(best_ann);
  //           fann_destroy(best_vnn);
  //         }
  //         best_ann = fann_copy(ann->getNeuralNet());
  //         best_vnn = fann_copy(vnn->getNeuralNet());
  //       };
  //
  // //       bib::Converger::min_stochastic_with_neg<>(iter, eval, save_best, 200, converge_precision, 0, 80, "ac_inter");
  //       vnn->copy(best_vnn);
  //       ann->copy(best_ann);
  //       fann_destroy(best_ann);
  //       fann_destroy(best_vnn);

  int k = 0;
  resetlooping = &k;
  for(k =0; k<10; k++)
    iter();

  //     if(episode>20)
  //     exit(1);
  //     LOG_DEBUG(ann->weight_l1_norm());

  double mmax = *std::max_element(scores_sum_q.begin(), scores_sum_q.end());
  uint index_best2 = 0;
  while(scores_sum_q[index_best2] < mmax)
    index_best2++;

  LOG_DEBUG("select " << index_best2 <<" with score " << mmax );
  delete ann;
  ann = new MLP(*proposed_policies[index_best2]);

  for(MLP* prop : proposed_policies) {
    delete prop;
  }

  return;

  if(episode >= 2) {

    std::vector<sample> traj(trajectory.size() < 60 ? trajectory.size() : 60);
    sample_transition(traj);

    double best_score=-50000.f;
    int index_best=-1;
    int index=0;

    for(MLP* prop : proposed_policies) {
#define MAXIMIN
#ifdef MAXIMIN //problem: s'eloigne de toutes les pols même la meilleure
      std::vector<double> score(old_policies.size(),0.f);

      for(auto sa : traj) {
        uint i2=0;
        std::vector<double>* pac = prop->computeOut(sa.s);
        shrink_actions(pac);
        for(auto old : old_policies) {
          std::vector<double>* oac = old.ann->computeOut(sa.s);
          shrink_actions(oac);
          score[i2] += bib::Utils::euclidien_dist(*pac, *oac, 2.);
          delete oac;
          i2++;
        }

        delete pac;
      }

      if(best_score < *std::min_element(score.begin(), score.end())) {
        best_score= *std::min_element(score.begin(), score.end());
        index_best=index;
      }

      index++;
#elif defined SCORER
      double score = 0;

      for(auto sa : traj) {
        uint i2=0;
        std::vector<double>* pac = prop->computeOut(sa.s);
        shrink_actions(pac);
        for(auto old : old_policies) {
          std::vector<double>* oac = old.ann->computeOut(sa.s);
          shrink_actions(oac);
          if(bib::Utils::euclidien_dist(*pac, *oac, 2.) > 0.0f)
            score += old.J/bib::Utils::euclidien_dist(*pac, *oac, 2.);
          else
            score += old.J;//bof bof, cannot penalize if only one ac is the same
          delete oac;
          i2++;
        }

        delete pac;
      }

      if(best_score < score) {
        best_score= score;
        index_best=index;
      }

      index++;
#else //closer to the best (might not be a good idea if the proposed pol is better than the current best)
      double score = 0.0f;

      for(auto sa : traj) {
        std::vector<double>* pac = prop->computeOut(sa.s);
        shrink_actions(pac);


        auto old = *old_policies.rbegin();
        std::vector<double>* oac = old.ann->computeOut(sa.s);
        shrink_actions(oac);
        score -= bib::Utils::euclidien_dist(*pac, *oac, 2.);
        delete oac;


        delete pac;
      }

      if(score <= 0.0f && score >= 0.0f)
        score=-50000;

      if(best_score < score) {
        best_score= score;
        index_best=index;
      }

      index++;
#endif
    }

    if(best_score <= 0.01 && best_score >= 0.0)
      LOG_DEBUG("should continue to search, i'm going to test the same pol as before");
    else
      LOG_DEBUG("select " << index_best <<" with score " << best_score );

    delete ann;
    ann = new MLP(*proposed_policies[index_best]);
    //     write_policy_file(andme++);

    for(MLP* prop : proposed_policies)
      delete prop;
  }
}
