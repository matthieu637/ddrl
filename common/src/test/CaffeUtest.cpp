#include <boost/iostreams/stream.hpp>

#include "gtest/gtest.h"
#include "nn/MLP.hpp"
#include "nn/DODevMLP.hpp"
#include "bib/Combinaison.hpp"

#include "boost/property_tree/ini_parser.hpp"
#include "boost/iostreams/stream.hpp"
#include "caffe/layers/developmental_layer.hpp"

double OR(double x1, double x2) {
  if (x1 == 1.f || x2 == 1.f)
    return 1.f;
  return -1.f;
}

double AND(double x1, double x2) {
  if (x1 == 1.f && x2 == 1.f)
    return 1.f;
  return -1.f;
}

TEST(MLP, ConsistentActivationFunction) {
  MLP nn(1, 1, {1}, 0.01f, 1, -1, 2, 0);

  EXPECT_DOUBLE_EQ(nn.number_of_parameters(), 4);
  double weights[] = {1,0,1,0};
  nn.copyWeightsFrom(weights, false);

  std::vector<double> sens(1);
  std::vector<double> ac(0);
  sens[0] = 1;

  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), tanh(1.));

  MLP nn2(1, 1, {1}, 0.01f, 1, -1, 1, 0);
  nn2.copyWeightsFrom(weights, false);

  EXPECT_DOUBLE_EQ(nn2.computeOutVF(sens, ac), 1);
  sens[0] = -1;
  EXPECT_GE(nn2.computeOutVF(sens, ac), -0.01 - 0.001);//leaky relu
  EXPECT_LE(nn2.computeOutVF(sens, ac), -0.01 + 0.001);//leaky relu

  double weights2[] = {0,0,0,5};
  nn2.copyWeightsFrom(weights2, false);

  EXPECT_DOUBLE_EQ(nn2.computeOutVF(sens, ac), 5.f);

  double weights3[] = {0.2,0.4,0.6, 0.8};
  nn.copyWeightsFrom(weights3, false);

  EXPECT_GE(nn.computeOutVF(sens, ac), (tanh((-1. * 0.2f + 0.4f)) * 0.6f + 0.8f) - 0.0001);
  EXPECT_LE(nn.computeOutVF(sens, ac), (tanh((-1. * 0.2f + 0.4f)) * 0.6f + 0.8f) + 0.0001);
}

TEST(MLP, LearnOpposite) {
  MLP nn(1, 1, {1}, 0.01f, 1, -1, 2, 0);
  double weights[] = {-1,0.,1. / tanh(1.), 0};
  nn.copyWeightsFrom(weights, false);

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> sens(1);
    std::vector<double> ac(0);
    sens[0] = x1;

    EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), out);
  }

  // Learn
  for (uint n = 0; n < 1000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> sens(1);
    std::vector<double> ac(0);
    sens[0] = x1;

    nn.learn(sens, ac, out);
  }

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> sens(1);
    std::vector<double> ac(0);
    sens[0] = x1;

    EXPECT_EQ(nn.computeOutVF(sens, ac), out);
  }
}

TEST(MLP, LearnAndOr) {
  MLP nn(3, 2, {10}, 0.01f, 1, -1, 2, 0);

  // Learn
  for (uint n = 0; n < 10000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;

    nn.learn(sens, ac, out);
  }

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.16);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.16);
  }
}


// 1 0.2
// 3 0.3
// 5 0.3
//
// Result:  y = 0.025 x + 1.916666667·10-1
//
// 2.166666667·10-1              1.666666667·10-2
// 2.666666667·10-1              3.333333333·10-2
// 3.166666667·10-1              1.666666667·10-2
// TEST(MLP, LearnLabelWeight) {

TEST(MLP, LearnNonLinearLabelWeight) {
  uint batch_size = 5004;
  MLP nn(1, 1, {4}, 0.005f, batch_size, -1, 2, 0, true);

  std::vector<double> sensors(batch_size);
  std::vector<double> actions(0);
  std::vector<double> qvalues(batch_size);
  std::vector<double> label_weight(batch_size);

  uint n = 0;
  auto iter = [&](const std::vector<double>& x) {
    sensors[n] = x[0];
    qvalues[n] = x[0];
    label_weight[n] = 1.f;
    n++;
  };

  auto iter2 = [&](const std::vector<double>& x) {
    sensors[n] = x[0];
    qvalues[n] = x[0]*x[0];
    label_weight[n] = 0.2f;
    n++;
  };

  bib::Combinaison::continuous<>(iter, 1, 0, 1, batch_size/2 );
  bib::Combinaison::continuous<>(iter2, 1, 0, 1, batch_size/2 );
  EXPECT_EQ(n, batch_size);

  nn.learn_batch_lw(sensors, actions, qvalues, label_weight, 300);

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01();

    double old_out = (x1 + x1*x1)/2.;
    double out = (x1 + 0.2*x1*x1)/1.2;

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(0);

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.1);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.1);

    if(x1 > 0.15 && x1 < 0.85)
      EXPECT_GE(fabs(nn.computeOutVF(sens, ac) - old_out), fabs(nn.computeOutVF(sens, ac) - out));
  }
}

TEST(MLP, LearnNonLinearLabelWeightWithNullImportance) {
  MLP nn(1, 1, {4}, 0.01f, 11*2, -1, 2, 0, true);

  std::vector<double> sensors(11*2);
  std::vector<double> actions(0);
  std::vector<double> qvalues(11*2);
  std::vector<double> label_weight(11*2);

  uint n = 0;
  auto iter = [&](const std::vector<double>& x) {
    sensors[n] = x[0];
    qvalues[n] = x[0];
    label_weight[n] = 0.0f;
    n++;
  };

  auto iter2 = [&](const std::vector<double>& x) {
    sensors[n] = x[0];
    qvalues[n] = x[0]*x[0];
    label_weight[n] = 0.8f;
    n++;
  };

  bib::Combinaison::continuous<>(iter, 1, 0, 1, 10);
  bib::Combinaison::continuous<>(iter2, 1, 0, 1, 10);

  nn.learn_batch_lw(sensors, actions, qvalues, label_weight, 500);

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01();

    double out = (0.0*x1 + 0.8*x1*x1)/(0.0+0.8);

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(0);

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.18);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.18);
  }
}

TEST(MLP, CaffeSaveLoad) {
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
//       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      MLP nn(2, 1, {50,10}, 0.005f, batch_size, -1, hidden_layer, batch_norm);

      std::vector<double> sensors(batch_size);
      std::vector<double> actions(batch_size);
      std::vector<double> qvalues(batch_size);

      for(uint forced = 0; forced < 1 ; forced++) {
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];
          actions[n] = x[1];

          if(bib::Utils::rand01() < 0.4) {
            sensors[n] = bib::Utils::rand01()*2 -1;
            actions[n] = bib::Utils::rand01()*2 -1;
          }
          qvalues[n] = -(actions[n]*actions[n]-(sensors[n]*sensors[n]));

          n++;
        };

        bib::Combinaison::continuous<>(iter, 2, -1, 1, sqrt(batch_size)-1);

        EXPECT_EQ(n, batch_size);
        nn.learn_batch(sensors, actions, qvalues, 50);
      }

      nn.save("CaffeSaveLoad.cache");
      MLP nn_test(nn, false);

      MLP nn2(2, 1, {50,10}, 0.005f, batch_size, -1, hidden_layer, batch_norm);
      nn2.load("CaffeSaveLoad.cache");

      MLP nn3(nn2, false);
      MLP nn4(nn2, true);//learning net

      for (uint n = 0; n < 100 ; n++) {
        double x0 = bib::Utils::rand01()*2 -1;
        double x1 = bib::Utils::rand01()*2 -1;

//         double out = -(x1*x1-(x0*x0));

        std::vector<double> sens(1);
        std::vector<double> ac(1);
        sens[0] = x0;
        ac[0]= x1;

//    This is not a test of performance
//      Some configuration of batch norm and hidden layer are very bad
//         EXPECT_GT(nn_test.computeOutVF(sens, ac), out - 0.3);
//         EXPECT_LT(nn_test.computeOutVF(sens, ac), out + 0.3);
//         
//         if(batch_norm == 0) {
// //        nn and nn2 are in learning phase so it's still learn batch norm
// //        don't test them during learning with batch norm
//           EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.21);
//           EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.21);
//         }

        //check that learning net are the same
        EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), nn2.computeOutVF(sens, ac));
        EXPECT_DOUBLE_EQ(nn2.computeOutVF(sens, ac), nn4.computeOutVF(sens, ac));

        //then check that testing net are the same
        EXPECT_DOUBLE_EQ(nn_test.computeOutVF(sens, ac), nn3.computeOutVF(sens, ac));

      }

    }
  }
}

TEST(MLP, CaffeCopyActor) {
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      std::vector<double> sensors(batch_size);

      MLP actor(1, {8}, 1, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      MLP actor_test(actor, true);//train
      MLP actor_test2(actor, false);//test
      MLP actor_test3(actor, true, ::caffe::Phase::TEST);//test
      MLP actor_test4(actor_test2, false);//test

      uint n = 0;
      auto iter = [&](const std::vector<double>& x) {
        sensors[n] = x[0];

        if(bib::Utils::rand01() < 0.4) {
          sensors[n] = bib::Utils::rand01()*2 -1;
        }
        n++;
      };

      bib::Combinaison::continuous<>(iter, 1, -1, 1, batch_size);

      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor_test.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs3 = actor_test2.computeOutBatch(sensors);
        auto all_actions_outputs4 = actor_test3.computeOutBatch(sensors);
        auto all_actions_outputs5 = actor_test4.computeOutBatch(sensors);
        for (uint i =0; i < batch_size; i++) {
          EXPECT_DOUBLE_EQ(all_actions_outputs->at(i), all_actions_outputs2->at(i));
          EXPECT_DOUBLE_EQ(all_actions_outputs4->at(i), all_actions_outputs3->at(i));
          EXPECT_DOUBLE_EQ(all_actions_outputs4->at(i), all_actions_outputs5->at(i));
          if(batch_norm == 0) //is not the same because batch norm is not learning
            EXPECT_DOUBLE_EQ(all_actions_outputs2->at(i), all_actions_outputs3->at(i));
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
        delete all_actions_outputs3;
        delete all_actions_outputs4;
        delete all_actions_outputs5;
      }
    }
  }
}

TEST(MLP, BackwardActor) {//valgrind test only
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <= 3 ; hidden_layer++) {
      LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      
      std::vector<double> sensors(batch_size);
      
      MLP actor(1, {8}, 1, 0.01f, batch_size, hidden_layer, 2, batch_norm);
      
      for(uint forced = 0; forced < 5 ; forced++) {
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];
          
          if(bib::Utils::rand01() < 0.4) {
            sensors[n] = bib::Utils::rand01()*2 -1;
          }
          n++;
        };
        
        bib::Combinaison::continuous<>(iter, 2, -1, 1, sqrt(batch_size)-1);
        
        auto all_actions_outputs = actor.computeOutBatch(sensors);
        
        //call to forward needed
        actor.actor_backward();
        delete all_actions_outputs;
        
        caffe::Blob<double> diff({400,1});
        all_actions_outputs = actor.computeOutBatch(sensors);
        const auto actor_actions_blob = actor.getNN()->blob_by_name(MLP::actions_blob_name);
        actor_actions_blob->ShareDiff(diff);
        actor.actor_backward();
        actor.getSolver()->ApplyUpdate();
        actor.getSolver()->set_iter(actor.getSolver()->iter() + 1);
        delete all_actions_outputs;
      }
    }
  }
}

TEST(MLP, BackwardCritic) {//valgrind test only
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <= 3 ; hidden_layer++) {
      LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      
      std::vector<double> sensors(batch_size);
      std::vector<double> actions(batch_size);
      
      MLP nn(2, 1, {50,10}, 0.001f, batch_size, -1, hidden_layer, batch_norm);
      
      for(uint forced = 0; forced < 5 ; forced++) {
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];
          actions[n] = x[1];

          if(bib::Utils::rand01() < 0.75) {
            sensors[n] = bib::Utils::rand01()*2 -1;
            actions[n] = bib::Utils::rand01()*2 -1;
          }
          n++;
        };

        bib::Combinaison::continuous<>(iter, 2, -1, 1, sqrt(batch_size)-1);

        auto all_q = nn.computeOutVFBatch(sensors, actions);
        
        //call to forward needed
        nn.critic_backward();
        delete all_q;
      }
      
    }
  }
}

TEST(MLP, OptimizeNNTroughGradientOfAnotherNN) {
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      MLP nn(2, 1, {50,10}, 0.001f, batch_size, -1, hidden_layer, batch_norm);//learning

      std::vector<double> sensors(batch_size);
      std::vector<double> actions(batch_size);
      std::vector<double> qvalues(batch_size);

      for(uint forced = 0; forced < 50 ; forced++) {
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];
          actions[n] = x[1];

          if(bib::Utils::rand01() < 0.75) {
            sensors[n] = bib::Utils::rand01()*2 -1;
            actions[n] = bib::Utils::rand01()*2 -1;
          }
          qvalues[n] = -(actions[n]*actions[n]-(sensors[n]*sensors[n])*actions[n]);

          n++;
        };

        bib::Combinaison::continuous<>(iter, 2, -1, 1, sqrt(batch_size)-1);

        EXPECT_EQ(n, batch_size);
        nn.learn_batch(sensors, actions, qvalues, 50);
      }

      MLP nn_test(nn, false);//testing
      MLP actor(1, {8}, 1, 0.01f, batch_size, hidden_layer, 2, batch_norm);
      MLP actor_test(actor, true);

      for(uint forced = 0; forced < 50 ; forced++) {
//         LOG_DEBUG("forced  "<< forced);
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];

          if(bib::Utils::rand01() < 0.4) {
            sensors[n] = bib::Utils::rand01()*2 -1;
          }
          n++;
        };

        bib::Combinaison::continuous<>(iter, 2, -1, 1, sqrt(batch_size)-1);

        nn.ZeroGradParameters();
        nn_test.ZeroGradParameters();

        auto all_actions_outputs = actor.computeOutBatch(sensors);
        auto all_actions_outputs2 = actor_test.computeOutBatch(sensors);
        if(forced == 0)
          for (uint i =0; i < batch_size; i++) {
            EXPECT_DOUBLE_EQ(all_actions_outputs->at(i), all_actions_outputs2->at(i));
          }

        delete nn.computeOutVFBatch(sensors, *all_actions_outputs);
        delete nn_test.computeOutVFBatch(sensors, *all_actions_outputs2);

        const auto q_values_blob = nn.getNN()->blob_by_name(MLP::q_values_blob_name);
        const auto q_values_blob2 = nn_test.getNN()->blob_by_name(MLP::q_values_blob_name);
        double* q_values_diff = q_values_blob->mutable_cpu_diff();
        double* q_values_diff2 = q_values_blob2->mutable_cpu_diff();
        for (uint i =0; i < batch_size; i++) {
          q_values_diff[q_values_blob->offset(i,0,0,0)] = -1.0f;
          q_values_diff2[q_values_blob2->offset(i,0,0,0)] = -1.0f;
        }
        nn.critic_backward();
        nn_test.critic_backward();
        const auto critic_action_blob = nn.getNN()->blob_by_name(MLP::actions_blob_name);
        const auto critic_action_blob2 = nn_test.getNN()->blob_by_name(MLP::actions_blob_name);

        const double* action_diff = critic_action_blob->cpu_diff();
        const double* action_diff2 = critic_action_blob2->cpu_diff();

        if(batch_norm == 0 && forced == 0)
          for (uint n = 0; n < batch_size; ++n) {
            for (uint h = 0; h < 1; ++h) {
              int offset = critic_action_blob->offset(n,0,h,0);
              int offset2 = critic_action_blob2->offset(n,0,h,0);
              EXPECT_DOUBLE_EQ(action_diff[offset], action_diff2[offset2]);
            }
          }

        // Transfer input-level diffs from Critic to Actor
        const auto actor_actions_blob = actor.getNN()->blob_by_name(MLP::actions_blob_name);
        actor_actions_blob->ShareDiff(*critic_action_blob);
        actor.actor_backward();
        actor.getSolver()->ApplyUpdate();
        actor.getSolver()->set_iter(actor.getSolver()->iter() + 1);

        const auto actor_actions_blob2 = actor_test.getNN()->blob_by_name(MLP::actions_blob_name);
        actor_actions_blob2->ShareDiff(*critic_action_blob2);
        actor_test.actor_backward();
        actor_test.getSolver()->ApplyUpdate();
        actor_test.getSolver()->set_iter(actor_test.getSolver()->iter() + 1);

        delete all_actions_outputs;
        delete all_actions_outputs2;
      }

      //       Test
      double ac1_error = 0;
      double ac2_error = 0;
      double q_error = 0;
      for (uint n = 0; n < 100 ; n++) {
        double x1 = bib::Utils::rand01()*2 -1;

        double out = (x1 * x1)/2;
        double qout = out*out-(x1*x1)*out;
        qout = - qout;

        std::vector<double> sens(1);
        sens[0] = x1;
        std::vector<double> ac(1);
        ac[0]= out;

        auto actor1_out = actor.computeOut(sens);
        auto actor2_out = actor_test.computeOut(sens);
        LOG_FILE("OptimizeNNTroughGradientOfAnotherNNFann.data", x1 << " " << actor1_out->at(0) << " " <<
        actor2_out->at(0)<< " " << out);
//clear all;close all;X=load('OptimizeNNTroughGradientOfAnotherNNFann.data');plot(X(:,1),X(:,2), '.');hold on;plot(X(:,1),X(:,3), 'r.');plot(X(:,1),X(:,4), 'go');
        ac1_error += fabs(out - actor1_out->at(0));
        ac2_error += fabs(out - actor2_out->at(0));
        q_error += fabs(nn.computeOutVF(sens, ac) - qout);
        
        delete actor1_out;
        delete actor2_out;
      }

      ac1_error /= 100.f;
      ac2_error /= 100.f;
      q_error /= 100.f;
      
      //if critic well learned
      if(q_error < 0.1){
        EXPECT_LT(ac1_error, 0.15);
        EXPECT_LT(ac2_error, 0.22);
      }

      LOG_DEBUG("bn " << batch_norm << " hiddenl " << hidden_layer << " " << ac1_error << " " << ac2_error << " " << q_error);
    }
  }
}

TEST(MLP, DevelopmentalLayer) {
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      std::vector<double> sensors(batch_size);
      
      MLP actor(1, {8}, 1, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      DODevMLP actor2(1, {8}, 1, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      std::string config="[devnn]\nst_scale=false\nst_probabilistic=0\nac_scale=false\nac_probabilistic=0\nst_control=\nac_control=\n";
      boost::property_tree::ptree properties;
      boost::iostreams::stream<boost::iostreams::array_source> stream(config.c_str(), config.size());
      boost::property_tree::ini_parser::read_ini(stream, properties);
      actor2.exploit(&properties, nullptr);

      EXPECT_GT(actor2.number_of_parameters(), actor.number_of_parameters());
      EXPECT_EQ(actor2.number_of_parameters(true) - 2, actor.number_of_parameters(true));
      uint n = 0;
      auto iter = [&](const std::vector<double>& x) {
        sensors[n] = x[0];
        
        if(bib::Utils::rand01() < 0.4) {
          sensors[n] = bib::Utils::rand01()*2 -1;
        }
        n++;
      };
      
      //copy param of actor to actor2
      double* weights = new double[actor.number_of_parameters(true)];
      actor.copyWeightsTo(weights, true);
      double* weights2 = new double[actor2.number_of_parameters(true)];
      actor2.copyWeightsTo(weights2, true);
      for(uint i=1;i< actor2.number_of_parameters(true)-2; i++)
        weights2[i]=weights[i-1];
      weights2[0]=1.f;
      weights2[actor2.number_of_parameters(true)-1]=1.f;
      actor2.copyWeightsFrom(weights2, true);
      bib::Combinaison::continuous<>(iter, 1, -1, 1, batch_size);
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
        for (uint i =0; i < batch_size; i++) {
          EXPECT_DOUBLE_EQ(all_actions_outputs->at(i), all_actions_outputs2->at(i));
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
      }
      
      weights2[actor2.number_of_parameters(true)-1] = 0.6f;
      actor2.copyWeightsFrom(weights2, true);
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
        uint fit=0;
        for (uint i =0; i < batch_size; i++) {
          if(all_actions_outputs->at(i) == all_actions_outputs2->at(i))
            fit++;
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
        double proba = (((float)fit)/((float)batch_size));
        EXPECT_GE(proba, 0.53);
        EXPECT_LE(proba, 0.67);
      }
      
      delete[] weights;
      delete[] weights2;
    }
  }
}

TEST(MLP, DevelopmentalLayerMoreDimension) {
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 343; //must be a power3 number
      std::vector<double> sensors(batch_size*3);
      
      MLP actor(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      DODevMLP actor2(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      std::string config="[devnn]\nst_scale=false\nst_probabilistic=0\nac_scale=false\nac_probabilistic=0\nst_control=\nac_control=None\n";
      boost::property_tree::ptree properties;
      boost::iostreams::stream<boost::iostreams::array_source> stream(config.c_str(), config.size());
      boost::property_tree::ini_parser::read_ini(stream, properties);
      actor2.exploit(&properties, nullptr);
      
      EXPECT_EQ(actor2.number_of_parameters(true) - 3 - 4, actor.number_of_parameters(true));
      uint n = 0;
      auto iter = [&](const std::vector<double>& x) {
        sensors[n] = x[0];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[1];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[2];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
      };
      
      //copy param of actor to actor2
      double* weights = new double[actor.number_of_parameters(true)];
      actor.copyWeightsTo(weights, true);
      double* weights2 = new double[actor2.number_of_parameters(true)];
      actor2.copyWeightsTo(weights2, true);
      for(uint i=0;i<3;i++)
        weights2[i]=1.f;
      for(uint i=3;i< actor.number_of_parameters(true); i++)
        weights2[i]=weights[i-3];
      for(uint i=actor.number_of_parameters(true); i < actor2.number_of_parameters(true)-3;i++)
        weights2[i+3]=1.f;
      actor2.copyWeightsFrom(weights2, true);
      bib::Combinaison::continuous<>(iter, 3, -1, 1, 6);
      CHECK_EQ(n, batch_size*3);
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
        for (uint i =0; i < batch_size; i++) {
          EXPECT_DOUBLE_EQ(all_actions_outputs->at(i), all_actions_outputs2->at(i));
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
      }
      
      weights2[actor2.number_of_parameters(true)-4] = 0.1f;
      weights2[actor2.number_of_parameters(true)-3] = 0.8f;
      weights2[actor2.number_of_parameters(true)-2] = 0.4f;
      weights2[actor2.number_of_parameters(true)-1] = 0.6f;
      actor2.copyWeightsFrom(weights2, true);
      
      delete[] weights;
      delete[] weights2;
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
        uint fit0=0;
        uint fit1=0;
        uint fit2=0;
        uint fit3=0;
        uint y=0;
        for (uint i =0; i < batch_size; i++) {
          for(uint j=0;j<4;j++){
            if(all_actions_outputs->at(y) == all_actions_outputs2->at(y)){
              if(j==0)
                fit0++;
              else if(j==1)
                fit1++;
              else if(j==2)
                fit2++;
              else 
                fit3++;
            }
            y++;
          }
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
        double proba0 = (((float)fit0)/((float)batch_size));
        double proba1 = (((float)fit1)/((float)batch_size));
        double proba2 = (((float)fit2)/((float)batch_size));
        double proba3 = (((float)fit3)/((float)batch_size));
        double std_=0.095;
        EXPECT_GE(proba0, 0.1-std_);
        EXPECT_LE(proba0, 0.1+std_);
        EXPECT_GE(proba1, 0.8-std_);
        EXPECT_LE(proba1, 0.8+std_);
        EXPECT_GE(proba2, 0.4-std_);
        EXPECT_LE(proba2, 0.4+std_);
        EXPECT_GE(proba3, 0.6-std_);
        EXPECT_LE(proba3, 0.6+std_);
      }
    }
  }
}

TEST(MLP, DevelopmentalLayerControlRestriction) {
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 343; //must be a power3 number
      std::vector<double> sensors(batch_size*3);
      
      MLP actor(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      DODevMLP actor2(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      std::string config="[devnn]\nst_scale=false\nst_probabilistic=0\nac_scale=false\nac_probabilistic=0\nst_control=1:2\nac_control=1:2\n";
      boost::property_tree::ptree properties;
      boost::iostreams::stream<boost::iostreams::array_source> stream(config.c_str(), config.size());
      boost::property_tree::ini_parser::read_ini(stream, properties);
      actor2.exploit(&properties, nullptr);
      
      EXPECT_EQ(actor2.number_of_parameters(true) - (3-1) - (4-2), actor.number_of_parameters(true));
      uint n = 0;
      auto iter = [&](const std::vector<double>& x) {
        sensors[n] = x[0];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[1];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[2];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
      };
      
      //copy param of actor to actor2
      double* weights = new double[actor.number_of_parameters(true)];
      actor.copyWeightsTo(weights, true);
      double* weights2 = new double[actor2.number_of_parameters(true)];
      actor2.copyWeightsTo(weights2, true);
      for(uint i=0;i<3-1;i++)
        weights2[i]=1.f;
      for(uint i=3-1;i< actor.number_of_parameters(true); i++)
        weights2[i]=weights[i-(3-1)];
      for(uint i=actor.number_of_parameters(true); i < actor2.number_of_parameters(true)-(3-1);i++)
        weights2[i+(3-1)]=1.f;
      actor2.copyWeightsFrom(weights2, true);
      bib::Combinaison::continuous<>(iter, 3, -1, 1, 6);
      CHECK_EQ(n, batch_size*3);
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
        for (uint i =0; i < batch_size; i++) {
          EXPECT_DOUBLE_EQ(all_actions_outputs->at(i), all_actions_outputs2->at(i));
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
      }
      
      weights2[actor2.number_of_parameters(true)-2] = 0.4f;
      weights2[actor2.number_of_parameters(true)-1] = 0.6f;
      actor2.copyWeightsFrom(weights2, true);
      
      delete[] weights;
      delete[] weights2;
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
        uint fit0=0;
        uint fit1=0;
        uint fit2=0;
        uint fit3=0;
        uint y=0;
        for (uint i =0; i < batch_size; i++) {
          for(uint j=0;j<4;j++){
            if(all_actions_outputs->at(y) == all_actions_outputs2->at(y)){
              if(j==0)
                fit0++;
              else if(j==1)
                fit1++;
              else if(j==2)
                fit2++;
              else 
                fit3++;
            }
            y++;
          }
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
        double proba0 = (((float)fit0)/((float)batch_size));
        double proba1 = (((float)fit1)/((float)batch_size));
        double proba2 = (((float)fit2)/((float)batch_size));
        double proba3 = (((float)fit3)/((float)batch_size));
        double std_=0.11;
        EXPECT_GE(proba0, 1.-std_);
        EXPECT_LE(proba0, 1.+std_);
        EXPECT_GE(proba1, 0.4-std_);
        EXPECT_LE(proba1, 0.4+std_);
        EXPECT_GE(proba2, 0.6-std_);
        EXPECT_LE(proba2, 0.6+std_);
        EXPECT_GE(proba3, 1.-std_);
        EXPECT_LE(proba3, 1.+std_);
      }
    }
  }
}


TEST(MLP, DevelopmentalLayerSharedParameters) { 
  std::vector<uint> batch_norms = {0,4,6,7,8,10,11,12,14,15};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 343; //must be a power3 number
      std::vector<double> sensors(batch_size*3);
      
      MLP actor2(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      DODevMLP actor(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      std::string config="[devnn]\nst_scale=false\nst_probabilistic=0\nac_scale=false\nac_probabilistic=0\nst_control=1:2\nac_control=1:2\n";
      boost::property_tree::ptree properties;
      boost::iostreams::stream<boost::iostreams::array_source> stream(config.c_str(), config.size());
      boost::property_tree::ini_parser::read_ini(stream, properties);
      actor.exploit(&properties, nullptr);
      
      MLP critic(4+3, 3, {10,5}, 0.005f, batch_size, -1, hidden_layer, batch_norm);
      DODevMLP critic2(4+3, 3, {10,5}, 0.005f, batch_size, -1, hidden_layer, batch_norm);
      critic2.exploit(&properties, (MLP*) &actor);
      DODevMLP critic3(critic2, true);
      
      EXPECT_EQ(critic.number_of_parameters(true), critic2.number_of_parameters(true)-4);
      EXPECT_EQ(critic2.number_of_parameters(true), critic3.number_of_parameters(true));
      
      uint n = 0;
      auto iter = [&](const std::vector<double>& x) {
        sensors[n] = x[0];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[1];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[2];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
      };
      
      if(batch_norm == 0){
        auto blob2 = actor.getNN()->layer_by_name("devnn_states")->blobs()[0];
        auto blob3 = critic2.getNN()->layer_by_name("devnn_states")->blobs()[0];
        
        EXPECT_EQ(blob3->count(), blob2->count());
        //careful to learnable_params order
        EXPECT_EQ(blob3->count(), critic2.getNN()->learnable_params()[1]->count());
        EXPECT_EQ(blob3->count(), actor.getNN()->learnable_params()[0]->count());
        for (int i =0;i<blob3->count();i++){
          EXPECT_EQ(blob3->cpu_data()[i], blob2->cpu_data()[i]);
          EXPECT_EQ(blob3->cpu_data()[i], critic2.getNN()->learnable_params()[1]->cpu_data()[i]);
          EXPECT_EQ(blob2->cpu_data()[i], actor.getNN()->learnable_params()[0]->cpu_data()[i]);
        }

        //copy param of actor to actor2
        double* weights = new double[actor.number_of_parameters(true)];
        actor.copyWeightsTo(weights, true);
        
        double* weights2 = new double[critic2.number_of_parameters(true)];
        critic2.copyWeightsTo(weights2, true);
        
        double* weights3 = new double[critic3.number_of_parameters(true)];
        critic3.copyWeightsTo(weights3, true);
        
        const double* weights4 = critic2.getNN()->learnable_params()[1]->cpu_data();
        const double* weights5 = actor.getNN()->learnable_params()[0]->cpu_data();
        
        ASSERT(critic2.getNN()->learnable_params()[1]->count(), 2);
        ASSERT(actor.getNN()->learnable_params()[0]->count(), 2);
        
        for(uint i=0;i<3-1;i++){
          EXPECT_DOUBLE_EQ(weights[i], weights5[i]);
          EXPECT_DOUBLE_EQ(weights4[i], weights2[i+2]);
          EXPECT_DOUBLE_EQ(weights[i], weights2[i+2]);
          EXPECT_DOUBLE_EQ(weights2[i], weights3[i]);//copy constructor for DODEVMLP
        }
//      change parameters of actor
        for(uint i=actor.number_of_parameters(true); i < actor2.number_of_parameters(true)-(3-1);i++)
          weights2[i+(3-1)]=1.f;
        actor.copyWeightsFrom(weights, true);
        
        for(uint i=0;i<3-1;i++){
          EXPECT_DOUBLE_EQ(weights[i], weights5[i]);
          EXPECT_DOUBLE_EQ(weights4[i], weights2[i+2]);
          EXPECT_DOUBLE_EQ(weights[i], weights2[i+2]);
          EXPECT_DOUBLE_EQ(weights2[i], weights3[i]);//copy constructor for DODEVMLP
        }
        
        delete[] weights;
        delete[] weights2;
        delete[] weights3;
      }
      bib::Combinaison::continuous<>(iter, 3, -1, 1, 6);
      CHECK_EQ(n, batch_size*3);
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//no dev layer
        auto q_v = critic2.computeOutVFBatch(sensors, *all_actions_outputs2);
        
        auto blob2 = actor.getNN()->layer_by_name("devnn_actions")->blobs()[0];
        auto blob3 = critic2.getNN()->layer_by_name("devnn_actions")->blobs()[0];
        auto blob33 = critic3.getNN()->layer_by_name("devnn_actions")->blobs()[0];
        
        ASSERT(blob3->count(), blob2->count());
        ASSERT(blob33->count(), blob2->count());
        for (int i =0;i<blob3->count();i++){
          EXPECT_EQ(blob3->cpu_data()[i], blob2->cpu_data()[i]);
          EXPECT_EQ(blob33->cpu_data()[i], blob2->cpu_data()[i]);
        }
        
        auto blob4 = actor.getNN()->layer_by_name("devnn_states")->blobs()[0];
        auto blob5 = critic2.getNN()->layer_by_name("devnn_states")->blobs()[0];
        auto blob55 = critic3.getNN()->layer_by_name("devnn_states")->blobs()[0];
        
        ASSERT(blob4->count(), blob5->count());
        ASSERT(blob55->count(), blob5->count());
        for (int i =0;i<blob4->count();i++){
          EXPECT_EQ(blob4->cpu_data()[i], blob5->cpu_data()[i]);
          EXPECT_EQ(blob55->cpu_data()[i], blob5->cpu_data()[i]);
        }
        
        delete q_v;
        delete all_actions_outputs;
        delete all_actions_outputs2;
      }
    }
  }
}

TEST(MLP, DevelopmentalLayerScalingControl) {
  std::vector<uint> batch_norms = {0};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 343; //must be a power3 number
      std::vector<double> sensors(batch_size*3);
      
      MLP actor(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      DODevMLP actor2(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      std::string config="[devnn]\nst_scale=true\nst_probabilistic=0\nac_scale=true\nac_probabilistic=0\nst_control=1:2\nac_control=1:2\n";
      boost::property_tree::ptree properties;
      boost::iostreams::stream<boost::iostreams::array_source> stream(config.c_str(), config.size());
      boost::property_tree::ini_parser::read_ini(stream, properties);
      actor2.exploit(&properties, nullptr);
      
      EXPECT_EQ(actor2.number_of_parameters(true) - (3-1) - (4-2), actor.number_of_parameters(true));
      uint n = 0;
      auto iter = [&](const std::vector<double>& x) {
        sensors[n] = x[0];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[1];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
        
        sensors[n] = x[2];
        if(bib::Utils::rand01() < 0.4)
          sensors[n] = bib::Utils::rand01()*2 -1;
        n++;
      };
      
      //copy param of actor to actor2
      double* weights = new double[actor.number_of_parameters(true)];
      actor.copyWeightsTo(weights, true);
      double* weights2 = new double[actor2.number_of_parameters(true)];
      actor2.copyWeightsTo(weights2, true);
      for(uint i=0;i<3-1;i++)
        weights2[i]=1.f;
      for(uint i=3-1;i< actor.number_of_parameters(true); i++)
        weights2[i]=weights[i-(3-1)];
      for(uint i=actor.number_of_parameters(true); i < actor2.number_of_parameters(true)-(3-1);i++)
        weights2[i+(3-1)]=1.f;
      actor2.copyWeightsFrom(weights2, true);
      bib::Combinaison::continuous<>(iter, 3, -1, 1, 6);
      CHECK_EQ(n, batch_size*3);
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
        for (uint i =0; i < batch_size; i++) {
          EXPECT_DOUBLE_EQ(all_actions_outputs->at(i), all_actions_outputs2->at(i));
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
      }
      
      weights2[actor2.number_of_parameters(true)-2] = 0.4f;
      weights2[actor2.number_of_parameters(true)-1] = 0.6f;
      actor2.copyWeightsFrom(weights2, true);
      
      delete[] weights;
      delete[] weights2;
      
      for(uint forced=0;forced<5;forced++){
        auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
        auto all_actions_outputs2 = actor2.computeOutBatch(sensors);//batch norm learns
 
        uint y=0;
        uint fit0=0;
        uint fit1=0;
        uint fit2=0;
        uint fit3=0;
        for (uint i =0; i < batch_size; i++) {
          for(uint j=0;j<4;j++){
            if(j==0 && all_actions_outputs->at(y) == all_actions_outputs2->at(y))
              fit0++;
            else if(j==1 && std::fabs(all_actions_outputs->at(y)*(1./0.4) - all_actions_outputs2->at(y)) <= 1e-10 )
              fit1++;
            else if(j==2 && std::fabs(all_actions_outputs->at(y)*(1./0.6) - all_actions_outputs2->at(y)) <= 1e-10 )
              fit2++;
            else if(all_actions_outputs->at(y) == all_actions_outputs2->at(y))
              fit3++;
            y++;
          }
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
        double proba0 = (((float)fit0)/((float)batch_size));
        double proba1 = (((float)fit1)/((float)batch_size));
        double proba2 = (((float)fit2)/((float)batch_size));
        double proba3 = (((float)fit3)/((float)batch_size));
        double std_=0.09;
        EXPECT_GE(proba0, 1.-std_);
        EXPECT_LE(proba0, 1.+std_);
        EXPECT_GE(proba1, 0.4-std_);
        EXPECT_LE(proba1, 0.4+std_);
        EXPECT_GE(proba2, 0.6-std_);
        EXPECT_LE(proba2, 0.6+std_);
        EXPECT_GE(proba3, 1.-std_);
        EXPECT_LE(proba3, 1.+std_);
      }
    }
  }
}

TEST(MLP, DevelopmentalLayerBackward) {
  std::vector<uint> batch_norms = {0};
  for(uint batch_norm : batch_norms)
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++)
      for(uint probabilistic = 0; probabilistic <=3 ; probabilistic++){
        uint batch_size = 343; //must be a power3 number
        std::vector<double> sensors(batch_size*3);
        
        DODevMLP actor(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
        std::string config="[devnn]\nst_scale=false\nst_probabilistic=";
        config += std::to_string(probabilistic);
        config += "\nac_scale=false\nac_probabilistic=";
        config += std::to_string(probabilistic);
        config += "\nst_control=1:2\nac_control=1:2\n";
        boost::property_tree::ptree properties;
        boost::iostreams::stream<boost::iostreams::array_source> stream(config.c_str(), config.size());
        boost::property_tree::ini_parser::read_ini(stream, properties);
        actor.exploit(&properties, nullptr);
        
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];
          if(bib::Utils::rand01() < 0.4)
            sensors[n] = bib::Utils::rand01()*2 -1;
          n++;
          
          sensors[n] = x[1];
          if(bib::Utils::rand01() < 0.4)
            sensors[n] = bib::Utils::rand01()*2 -1;
          n++;
          
          sensors[n] = x[2];
          if(bib::Utils::rand01() < 0.4)
            sensors[n] = bib::Utils::rand01()*2 -1;
          n++;
        };
        
        bib::Combinaison::continuous<>(iter, 3, -1, 1, 6);
        CHECK_EQ(n, batch_size*3);
        
        for(uint forced=0;forced<5;forced++){
          auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
          actor.actor_backward();
          delete all_actions_outputs;
        }
      }
}


TEST(MLP, DevelopmentalLayerBackwardDiffCompute) {
  std::vector<uint> batch_norms = {0};
  for(uint batch_norm : batch_norms)
    for(uint hidden_layer = 1; hidden_layer <=3 ; hidden_layer++)
      for(uint probabilistic = 0; probabilistic <=3 ; probabilistic++){
        uint batch_size = 343; //must be a power3 number
        std::vector<double> sensors(batch_size*3);
        
        DODevMLP actor(3, {8}, 4, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
        std::string config="[devnn]\ncompute_diff_backward=true\nst_scale=false\nst_probabilistic=";
        config += std::to_string(probabilistic);
        config += "\nac_scale=false\nac_probabilistic=";
        config += std::to_string(probabilistic);
        config += "\nst_control=1:2\nac_control=1:2\n";
        boost::property_tree::ptree properties;
        boost::iostreams::stream<boost::iostreams::array_source> stream(config.c_str(), config.size());
        boost::property_tree::ini_parser::read_ini(stream, properties);
        actor.exploit(&properties, nullptr);
        
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];
          if(bib::Utils::rand01() < 0.4)
            sensors[n] = bib::Utils::rand01()*2 -1;
          n++;
          
          sensors[n] = x[1];
          if(bib::Utils::rand01() < 0.4)
            sensors[n] = bib::Utils::rand01()*2 -1;
          n++;
          
          sensors[n] = x[2];
          if(bib::Utils::rand01() < 0.4)
            sensors[n] = bib::Utils::rand01()*2 -1;
          n++;
        };
        
        bib::Combinaison::continuous<>(iter, 3, -1, 1, 6);
        CHECK_EQ(n, batch_size*3);
        
        for(uint forced=0;forced<5;forced++){
          auto all_actions_outputs = actor.computeOutBatch(sensors);//batch norm learns
          actor.actor_backward();
          delete all_actions_outputs;
        }
      }
}
