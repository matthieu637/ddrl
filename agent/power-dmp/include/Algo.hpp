#ifndef ALGO_HPP
#define ALGO_HPP

#include <eigen3/Eigen/Core>
#include "Kernel.hpp"
#include "Config.hpp"
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <fstream>
#include <bitset>
#include <iomanip>

class Algo
{
    public:

        struct Param
        {
            std::vector<Eigen::MatrixXf*> weights;
            std::vector<Eigen::MatrixXf*> variance;
            unsigned int episode;
        };
        /** Default constructor */
        Algo();
        Algo(Config*);
        ~Algo() {

            for(unsigned int i=0;i<best_params.size();i++){
                for(unsigned int j=0;j<(*best_params[i].second).weights.size();j++){
                    delete (*best_params[i].second).weights[j];
                }
                for(unsigned int j=0;j<(*best_params[i].second).variance.size();j++){
                    delete (*best_params[i].second).variance[j];
                }
                delete best_params[i].second;
            }

        }
        void computeNewWeights();
        std::vector<double> getNextActions(std::vector<double> states);
        void addReward(double const _reward );
       void setPointeurIteration(unsigned int* _iter);
       void setPointeurEpisode(unsigned int* _iter);
       void setPointeurConfig(Config* _config);
        Eigen::VectorXf normalDistribution(unsigned int size);
      void save(const std::string& path);
      void load(const std::string& path);
    protected:
    private:
        unsigned int n_states_per_kernels;
        unsigned int n_kernels;
        std::vector<Kernel> kernels;
        std::vector<Eigen::MatrixXf> variances;
        unsigned int *iter;
        unsigned int *episode;
        Param* current_param;
        Param* current_param_episode;
        unsigned int n_weights;
        std::vector< std::pair<double,Param*>> params_episode;
        std::vector< std::pair<double,Param*>> best_params;
        const double PI = 3.14159265358979f;
        Config* config;
};

#endif // ALGO_HPP
