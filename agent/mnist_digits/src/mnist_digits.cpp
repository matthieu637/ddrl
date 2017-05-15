#include <glog/logging.h>
#include <gflags/gflags.h>
#include <caffe/caffe.hpp>
#include "nn/MLP.hpp"
#include <iostream>
#include <vector>
#include <unistd.h>
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/ini_parser.hpp"

#include "bib/IniParser.hpp"
#include "bib/Logger.hpp"
#include "bib/Assert.hpp"
#include "mnist_extraction.hpp"


int main(int argc, char **argv) {
    
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();  
  
  // Data extraction
  vector<vector<double>> ar;
  vector<double> lbl;
  read_mnist_img(10000,784,ar);
  read_mnist_lbl(lbl);

  
  // Parameters
  int data_size = 784;   
  int label_size = 10;
  int training_examples = 1000;
  int input_size = 784;
  vector<uint> hiddens = {30};
  unsigned int motors = 10;
  double alpha = 0.001;
  uint kMinibatchSize = 10;
  int batchNum = training_examples / kMinibatchSize;
  uint hidden_layer_type = 2;
  uint last_layer_type = 0;
  /*
   *  0 : InnerProduct
   *  1 : LRelu
   *  2 : Tanh
   *  3 : Relu
   */
  uint batch_norm = 0;
  bool loss_layer = true;
  
  
  vector<vector<double>> training_data_batches = {};
  training_data_batches.resize(batchNum);
  vector<vector<double>> training_label_batches = {};
  training_label_batches.resize(batchNum);
  
  for(int i = 0; i < batchNum ; i++) {
    training_data_batches[i].resize(kMinibatchSize * data_size);
    training_label_batches[i].resize(kMinibatchSize * label_size,0);
    for(uint k = 0; k < kMinibatchSize; k++) {
      for(int j = 0; j < data_size; j++) {
	training_data_batches[i][k*data_size + j] = ar[i*kMinibatchSize + k][j];
      }     
      training_label_batches[i][k*label_size + lbl[i*kMinibatchSize + k]] = 1;      
    }           
  }
  
  print_mnist_img(450, ar);
  for (int i = 0; i < label_size ; i++) {
    cout << training_label_batches[45][i];
  }
  
  
  
  boost::property_tree::ptree properties;
  boost::property_tree::ini_parser::read_ini("config.ini", properties);

  double lr = properties.get<double>("simulation.learning_rate");
  hiddens = * bib::to_array<uint>(properties.get<std::string>("simulation.hiddens"));
  alpha = lr;
  
  // Network initialisation
  MLP my_network(input_size, hiddens, motors, alpha, kMinibatchSize, hidden_layer_type, last_layer_type, batch_norm, loss_layer);
  
  // Trainning
  for(uint i=0;i<100;i++){
   
   my_network.learn(training_data_batches[0], training_label_batches[0]);
   LOG_DEBUG(my_network.error() << " " << my_network.weight_l1_norm());
   LOG_FILE("learning.data", i << " " << my_network.error() << " " << my_network.weight_l1_norm());
  }
  
  my_network.save("testsv");
  
  
  
  google::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  google::protobuf::ShutdownProtobufLibrary();
  
  LOG_FILE("time_elapsed", "");
  
  return 0;
}
