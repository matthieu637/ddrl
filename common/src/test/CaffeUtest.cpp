
#include "gtest/gtest.h"
#include "nn/MLP.hpp"
#include "bib/Combinaison.hpp"

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
  nn.copyWeightsFrom(weights);

  std::vector<double> sens(1);
  std::vector<double> ac(0);
  sens[0] = 1;

  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), tanh(1.));

  MLP nn2(1, 1, {1}, 0.01f, 1, -1, 1, 0);
  nn2.copyWeightsFrom(weights);

  EXPECT_DOUBLE_EQ(nn2.computeOutVF(sens, ac), 1);
  sens[0] = -1;
  EXPECT_GE(nn2.computeOutVF(sens, ac), -0.01 - 0.001);//leaky relu
  EXPECT_LE(nn2.computeOutVF(sens, ac), -0.01 + 0.001);//leaky relu

  double weights2[] = {0,0,0,5};
  nn2.copyWeightsFrom(weights2);

  EXPECT_DOUBLE_EQ(nn2.computeOutVF(sens, ac), 5.f);

  double weights3[] = {0.2,0.4,0.6, 0.8};
  nn.copyWeightsFrom(weights3);

  EXPECT_GE(nn.computeOutVF(sens, ac), (tanh((-1. * 0.2f + 0.4f)) * 0.6f + 0.8f) - 0.0001);
  EXPECT_LE(nn.computeOutVF(sens, ac), (tanh((-1. * 0.2f + 0.4f)) * 0.6f + 0.8f) + 0.0001);
}

TEST(MLP, LearnOpposite) {
  MLP nn(1, 1, {1}, 0.01f, 1, -1, 2, 0);
  double weights[] = {-1,0.,1. / tanh(1.), 0};
  nn.copyWeightsFrom(weights);

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

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.15);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.15);
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
  std::vector<uint> batch_norms = {0, 1, 3};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=2 ; hidden_layer++) {
//       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      MLP nn(2, 1, {50,10}, 0.005f, batch_size, -1, hidden_layer, batch_norm);

      std::vector<double> sensors(batch_size);
      std::vector<double> actions(batch_size);
      std::vector<double> qvalues(batch_size);

      for(uint forced = 0; forced < 150 ; forced++) {
        uint n = 0;
        auto iter = [&](const std::vector<double>& x) {
          sensors[n] = x[0];
          actions[n] = x[1];

          if(bib::Utils::rand01() < 0.4) {
            sensors[n] = bib::Utils::rand01()*2 -1;
            actions[n] = bib::Utils::rand01()*2 -1;
          }
//       qvalues[n] = -(actions[n]*actions[n]-(sensors[n]*sensors[n])*actions[n]);
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

//     double out = -(x1*x1-(x0*x0)*x1);
        double out = -(x1*x1-(x0*x0));

        std::vector<double> sens(1);
        std::vector<double> ac(1);
        sens[0] = x0;
        ac[0]= x1;


        EXPECT_GT(nn_test.computeOutVF(sens, ac), out - 0.25);
        EXPECT_LT(nn_test.computeOutVF(sens, ac), out + 0.25);
        if(batch_norm == 0) {
//        nn and nn2 are in learning phase so it's still learn batch norm
//        don't test them during learning with batch norm
          EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.21);
          EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.21);
        }

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
  std::vector<uint> batch_norms = {0, 1, 3};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=2 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      std::vector<double> sensors(batch_size);

      MLP actor(1, {8}, 1, 0.01f, batch_size, hidden_layer, 0, batch_norm);//train
      MLP actor_test(actor, true);//train
      MLP actor_test2(actor, false);//test
      MLP actor_test3(actor, true, ::caffe::Phase::TEST);//test

      for(uint forced = 0; forced < 150 ; forced++) {
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
        auto all_actions_outputs2 = actor_test.computeOutBatch(sensors);
        auto all_actions_outputs3 = actor_test2.computeOutBatch(sensors);
        auto all_actions_outputs4 = actor_test3.computeOutBatch(sensors);
        for (uint i =0; i < batch_size; i++) {
          EXPECT_DOUBLE_EQ(all_actions_outputs->at(i), all_actions_outputs2->at(i));
          if(batch_norm == 0) //is not the same because batch norm is not learning
            EXPECT_DOUBLE_EQ(all_actions_outputs2->at(i), all_actions_outputs3->at(i));
          EXPECT_DOUBLE_EQ(all_actions_outputs4->at(i), all_actions_outputs3->at(i));
        }
        delete all_actions_outputs;
        delete all_actions_outputs2;
        delete all_actions_outputs3;
        delete all_actions_outputs4;
      }
    }
  }
}

TEST(MLP, OptimizeNNTroughGradientOfAnotherNN) {
  std::vector<uint> batch_norms = {0, 1, 3};
  for(uint batch_norm : batch_norms) {
    for(uint hidden_layer = 1; hidden_layer <=2 ; hidden_layer++) {
      //       LOG_DEBUG("bn " << batch_norm << " " << hidden_layer);
      uint batch_size = 400; //must be a squared number
      MLP nn(2, 1, {50,10}, 0.001f, batch_size, -1, hidden_layer, batch_norm);//learning

      std::vector<double> sensors(batch_size);
      std::vector<double> actions(batch_size);
      std::vector<double> qvalues(batch_size);

      for(uint forced = 0; forced < 500 ; forced++) {
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

      for(uint forced = 0; forced < 150 ; forced++) {
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
        nn.getNN()->BackwardFrom(nn.GetLayerIndex(MLP::q_values_layer_name));
        nn_test.getNN()->BackwardFrom(nn_test.GetLayerIndex(MLP::q_values_layer_name));
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
        actor.getNN()->BackwardFrom(actor.GetLayerIndex("action_layer"));
        actor.getSolver()->ApplyUpdate();
        actor.getSolver()->set_iter(actor.getSolver()->iter() + 1);

        const auto actor_actions_blob2 = actor_test.getNN()->blob_by_name(MLP::actions_blob_name);
        actor_actions_blob2->ShareDiff(*critic_action_blob2);
        actor_test.getNN()->BackwardFrom(actor_test.GetLayerIndex("action_layer"));
        actor_test.getSolver()->ApplyUpdate();
        actor_test.getSolver()->set_iter(actor_test.getSolver()->iter() + 1);

        delete all_actions_outputs;
        delete all_actions_outputs2;
      }

      //       Test
      double ac1_error = 0;
      double ac2_error = 0;
      for (uint n = 0; n < 100 ; n++) {
        double x1 = bib::Utils::rand01()*2 -1;

        double out = (x1 * x1)/2;
        double qout = out*out-(x1*x1)*out;
        qout = - qout;

        std::vector<double> sens(1);
        sens[0] = x1;
        std::vector<double> ac(1);
        ac[0]= out;

        if(fabs(nn.computeOutVF(sens, ac) - qout) > 0.1)
          break;

        EXPECT_GT(nn.computeOutVF(sens, ac), qout - 0.2);
        EXPECT_LT(nn.computeOutVF(sens, ac), qout + 0.2);

        std::vector<double> * ac_return = nullptr;

        if(batch_norm != 0) {
          ac_return = actor_test.computeOut(sens);
          EXPECT_GT(ac_return->at(0), out - 0.2);
          EXPECT_LT(ac_return->at(0), out + 0.2);
          delete ac_return;
        } else if(hidden_layer != 2){
          ac_return = actor.computeOut(sens);
          EXPECT_GT(ac_return->at(0), out - 0.2);
          EXPECT_LT(ac_return->at(0), out + 0.2);
          delete ac_return;
        }

        LOG_FILE("OptimizeNNTroughGradientOfAnotherNNFann.data", x1 << " " << actor.computeOut(sens)->at(0) << " " <<
                 actor_test.computeOut(sens)->at(0)<< " " << out);
//clear all;close all;X=load('OptimizeNNTroughGradientOfAnotherNNFann.data');plot(X(:,1),X(:,2), '.');hold on;plot(X(:,1),X(:,3), 'r.');plot(X(:,1),X(:,4), 'go');
        ac1_error += fabs(out - actor.computeOut(sens)->at(0));
        ac2_error += fabs(out - actor_test.computeOut(sens)->at(0));
      }

      ac1_error /= 100;
      ac2_error /= 100;

//       LOG_DEBUG("bn " << batch_norm << " hiddenl " << hidden_layer << " " << ac1_error << " " << ac2_error);
//       exit(1);
    }
  }
}


