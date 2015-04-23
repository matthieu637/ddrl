#include "PowerAg.hpp"
#include <random>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;
using namespace Eigen;

PowerAg::PowerAg(unsigned int nb_motors, unsigned int nb_sensors) :  kernel(4, 2), actuator(nb_motors) {

  n_kernels_per_dim = 4;
  n_dims = 2;
  n_kernels = std::pow(n_kernels_per_dim, n_dims*2);
  n_iter = 5000;
  iter = 0;
  //kernel=_kernel;
  param = MatrixXf::Zero(n_kernels,n_iter+1);
  variance = MatrixXf::Zero(n_kernels,n_iter+1);
  param.col(0) = VectorXf::Zero(n_kernels).transpose();
  const string path("rezre");
  //load(path);
  //param.col(0) = VectorXf::Zero(n_kernels).transpose();
  //param.col(0) = normalDistribution(n_kernels).transpose()*0.1;
  kernel.setWeights(param.col(0));
  current_param=param.col(0);
  variance.col(0) = VectorXf::Constant(n_kernels,0.1).transpose();
  variance.col(1) = VectorXf::Constant(n_kernels,0.1).transpose();
  }

const vector<float>& PowerAg::run(float _reward, const vector<float>& sensors, bool learning, bool goal_reached) {
if(goal_reached)
    cout << "GOAL REACHED" << endl;
   /* for (unsigned int i = 0; i < actuator.size(); i++)
      actuator[i] = bib::Utils::randin(-6, 6);*/
      //float r = pow(PI-abs(sensors[0]),2)/pow(PI,2)+pow(PI-abs(sensors[0]+sensors[2]),2)/pow(PI,2)+0.1*pow(actuator[0],2)/pow(0.5,2);
      //float r = pow(-1-cos(sensors[0]+sensors[2]),2)/4+pow(PI-abs(sensors[0]+sensors[2]),2)/pow(PI,2);
      //cout << sensors[0] << endl;

      //cout << pas << endl;
      //cout << rewards[iter] << endl;
      actuator[0]=kernel.getValue(sensors);
      //float r = pow(1+cos(sensors[0]),2)+pow(1+cos(sensors[0]+sensors[2]),2)+0.001*pow(actuator[0],2);
      float y1 = -cos(sensors[0]);
      float y2 = y1 - cos(sensors[0]+sensors[2]);
      //cout << sensors[3] << endl;
      float Ep = 10 * y1 + 10 * y2;
      //float r = pow(30-Ep,2)/pow(60,2)+0.005*pow(actuator[0],2);
      float r = pow(30-Ep,2)/pow(60,2)+pow(sensors[2],2)/pow(PI,2)+0.06*pow(actuator[0],2);
      //float r = pow(30-Ep,2)/pow(60,2)+pow(sensors[2],2)/pow(PI,2);
      //float r = pow(2+y2,2)/pow(2,2);
/*if(goal_reached)
    r+=100;*/
      //cout << rewards[iter] << endl;
      rewards[iter] += exp(-pow(r,0.5));
      //rewards[iter] += r;
      if(goal_reached)
        rewards[iter]+=5;
      //cout << rewards[iter] << endl;
      //actuator[i] = 0;
      //Energie +=sensors[3]*actuator[0]*0.005;
      /*if(sensors[3]>0){
        actuator[0]=1/(sensors[3]+0.001);
      } else {
        actuator[0]=1/(sensors[3]+0.001);
      }*/
      //cout << "sensor0" << sensors[0] << endl;
      //cout << "sensor1" << sensors[1] << endl;

      //cout << sensors[0] << endl;
      /*cout << "Energie : " << Energie << endl;*/
      pas++;
    return actuator;
  }

  void PowerAg::start_episode(const std::vector<float>& sensors) {

    pas=0;
    rewards.push_back(0);

Eigen::VectorXf param_nom = VectorXf::Zero(n_kernels).transpose();
Eigen::VectorXf param_dnom = VectorXf::Zero(n_kernels).transpose();
Eigen::VectorXf temp_W(n_kernels);
Eigen::VectorXf temp_explore(n_kernels);

if(iter>0){
int n_items = (iter<10)?iter:10;
for(int i=0;i<n_items;i++){

    int j = s_Return[s_Return.size()-1-i].second;

//    cout << j << endl;
//    cout << rewards[j] << endl;
    temp_W = variance.col(j).array().inverse();
    temp_explore = param.col(j)-param.col(iter-1);
    param_nom = param_nom.array() + temp_W.array()*temp_explore.array()*rewards[j];
    param_dnom = param_dnom.array() + temp_W.array()*rewards[j];
//cout << param_dnom << endl;
}
//cout << param.col(1).array() << endl;
//param.col(iter) = param.col(iter-1).array() ;
param.col(iter) = param.col(iter-1).array() + param_nom.array()/(param_dnom.array()+0.0000000001);
    if(iter>1){
        Eigen::VectorXf var_nom = VectorXf::Zero(n_kernels).transpose();
        float var_dnom = 0;
        n_items = (iter<100)?iter:100;
        for(int i=0;i<n_items;i++){
            int j = s_Return[s_Return.size()-1-i].second;
            temp_explore = param.col(j)-param.col(iter-1);
            var_nom = var_nom.array() + temp_explore.array().square()*rewards[j];
            var_dnom = var_dnom + rewards[j];
        }
        variance.col(iter) = var_nom.array() / var_dnom;
       variance.col(iter)=variance.col(iter).cwiseMin(10*variance.col(0));
        variance.col(iter)=variance.col(iter).cwiseMax(0.1*variance.col(0));

           // cout << variance.col(iter) << endl;
    }
    //std::ptr_fun(normalDistribution);
    Eigen::VectorXf noise = normalDistribution(n_kernels);
    //noise = noise.unaryExpr(ptr_fun(normalDistribution));
    //cout << noise << endl;
    param.col(iter) = param.col(iter).array() + variance.col(iter).array().sqrt()*noise.array();
  }
  //cout << param.col(iter) << endl;
kernel.setWeights(param.col(iter));
  }
  Eigen::VectorXf PowerAg::normalDistribution(int _size){
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0,1.0);
    //distribution.reset();
    Eigen::VectorXf retour(_size);
    for(unsigned int i=0;i<_size;i++)
        retour(i) = distribution(generator);
    //cout << retour << endl;
    return retour;
  }
  void PowerAg::end_episode() {
    //cout << rewards.size() << endl;
    //cout << rewards[iter] << endl;
    rewards[iter]=rewards[iter]/pas;
//    cout << iter << endl;
//    cout << rewards[iter] << endl;
    s_Return.push_back( pair<double,int> (rewards[iter],iter));
    sort(s_Return.begin(), s_Return.end());
    //for(int i=0; i < iter; i++){}
    iter++;

        //cout << s_Return[i].first << endl;
  }

  void PowerAg::save(const std::string& path) {
      MatrixXf param_to_save = MatrixXf::Zero(n_kernels,10);
        for(int i=0;i<10;i++){
            int j = s_Return[s_Return.size()-1-i].second;
            param_to_save.col(i) = param.col(j);
        }
        vector<float> p(n_kernels);
        for(int i=0;i<n_kernels;i++)
            p[i]=param_to_save(i,0);
       ofstream fichier(path, ios::out | ios::trunc);  // ouverture en écriture avec effacement du fichier ouvert
        LOG_DEBUG("save " << 3);
        if(fichier)
        {
            for(int i=0;i<n_kernels;i++)
                fichier << p[i] << endl;
                fichier.close();
        }
        else
                cerr << "Impossible d'ouvrir le fichier !" << endl;

  }

  void PowerAg::load(const std::string& path) {
        ifstream fichier(path, ios::in);  // on ouvre en lecture
cout << "load" << endl;
        if(fichier)  // si l'ouverture a fonctionné
        {
                string ligne;
                int i(0);
                while(getline(fichier, ligne))  // tant que l'on peut mettre la ligne dans "contenu"
                {
                        param(i,0) = ::atof(ligne.c_str());
                        i++;
                }

                fichier.close();
        }
        else
                cerr << "Impossible d'ouvrir le fichier !" << endl;

  }

