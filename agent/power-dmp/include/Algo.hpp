#ifndef ALGO_HPP
#define ALGO_HPP

#include <eigen3/Eigen/Core>
#include "Kernel.hpp"
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
        Algo(unsigned int _n_kernels);
        ~Algo() {
            delete iter;
            delete current_param;
        }
        void computeNewWeights();
        std::vector<float> getNextActions(std::vector<float> states);
        void addReward(float const _reward );
       void setPointeurIteration(unsigned int* _iter);
       void setPointeurEpisode(unsigned int* _iter);
        Eigen::VectorXf normalDistribution(unsigned int size);
      void save(const std::string& path);
      void load(const std::string& path);
    protected:
    private:
        unsigned int n_states_per_kernels;
        unsigned int n_kernels;
     // vector<Eigen::MatrixXf> param;
      //Eigen::MatrixXf variance;
        std::vector<Kernel> kernels;
        std::vector<Eigen::MatrixXf> variances;
        unsigned int *iter;
        unsigned int *episode;
        Param* current_param;
        Param* current_param_episode;
        unsigned int n_weights;
        std::vector< std::pair<float,Param*>> params_episode;
        std::vector< std::pair<float,Param*>> best_params;
        const float PI = 3.14159265358979f;
};

#endif // ALGO_HPP
