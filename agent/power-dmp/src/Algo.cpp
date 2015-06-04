#include "Algo.hpp"

using namespace std;
using namespace Eigen;

Algo::Algo() {}
Algo::Algo(Config* _config) : config(_config)
{
    //unsigned int const n_motors_per_dim = 2;
    //unsigned int const n_states_per_kernels=4;

    n_weights = std::pow((*config).n_kernels_per_dim, (*config).n_states_per_kernels);

    /*n_weights=1;
    for(int i=0;i<nDims;i++){
        m_nKernelsPerDim[i]=nKernelsPerDim;
        m_nKernels*=m_nKernelsPerDim[i];
    }*/

    Param *new_param = new Param();
    for(unsigned int i=0;i<(*config).n_motors;i++){

        unsigned int const n_var = 2;
        MatrixXf variance_init = MatrixXf::Zero(n_weights,n_var);
        variance_init.col(0) = VectorXf::Constant(n_weights,(*config).var_init).transpose();
        variance_init.col(1) = VectorXf::Constant(n_weights,(*config).var_init).transpose();
        MatrixXf *weights = new MatrixXf(MatrixXf::Zero(n_weights,1));
        MatrixXf *variance = new MatrixXf(variance_init.col(0));
    //*weights=variances[i_kernel].col(*iter).array().sqrt()*noise.array();
        (*new_param).weights.push_back(weights);
        (*new_param).variance.push_back(variance);
        Kernel kernel((*config).n_kernels_per_dim,(*config).n_states_per_kernels);
        variances.push_back(variance_init);
        kernels.push_back(kernel);
    }
    current_param=new_param;
}
void Algo::setPointeurConfig(Config* _config){
    config=_config;
}
void Algo::setPointeurIteration(unsigned int* _iter){
    iter=_iter;
}
void Algo::setPointeurEpisode(unsigned int* _iter){
    episode=_iter;
}
void Algo::addReward(float const _reward){
    if(*episode>1){
        (*current_param_episode).episode = *iter;
        params_episode.push_back( pair<float,Param*> (_reward,current_param_episode));
        sort(params_episode.begin(), params_episode.end());
    }
    if(*episode>1 && ((*iter)+1)%(*config).n_instances==0){
        best_params.push_back( pair<float,Param*> (params_episode[params_episode.size()-1].first,params_episode[params_episode.size()-1].second));
        sort(best_params.begin(), best_params.end());
        params_episode.clear();
        if(*episode>100)
            best_params.erase(best_params.begin());
    } else if(*episode==1 && ((*iter)+1)%(*config).n_instances==0){
        (*current_param).episode = *iter;
        best_params.push_back( pair<float,Param*> (_reward,current_param));
    }
    /*if(*episode>100){
        //delete best_params[0].second;
        //delete p;
        for(int i=0;i<(*best_params[0].second).weights.size();i++)
            delete (*best_params[0].second).weights[i];
        for(int i=0;i<(*best_params[0].second).variance.size();i++)
            delete (*best_params[0].second).variance[i];

    }*/
}
void Algo::computeNewWeights(){

    Param *new_param = new Param(*current_param);

    for(unsigned int i_kernel=0;i_kernel<(*config).n_motors;i_kernel++){

        Eigen::VectorXf param_nom = VectorXf::Zero(n_weights).transpose();
        Eigen::VectorXf param_dnom = VectorXf::Zero(n_weights).transpose();
        Eigen::VectorXf temp_W(n_weights);

        Eigen::VectorXf temp_explore(n_weights);

        //if(*episode>1&&( ((((*episode/10+i_kernel))%(n_motors))!=0)||*episode%10==0)){
        //if(*iter>1&&( ((((*iter/10+i_kernel))%(n_motors))!=0)||(*iter%10==0))){
        //if(*iter>1&&(i_kernel==(n_motors-1)||(*iter%10==0))){
        if((*episode>1&&(*config).n_motors<3)||(*episode>1&&( ((((*episode/10+i_kernel))%((*config).n_motors))!=0)||*episode%10==0))){

        const int m = *iter%((*config).n_instances)*4+2;
        //const int m = (*config).elite;
        int n_items = (*episode-1<m)?*episode-1:m;

        for(int i=0;i<n_items;i++){
            Param* par = best_params[best_params.size()-1-i].second;
            float reward_j = best_params[best_params.size()-1-i].first;
            /*if(i_kernel==((*config).n_motors-1) && i<11){
            cout << (*par).episode << endl;
            cout << reward_j << endl;
            }*/
            temp_W = (*par).variance[i_kernel]->col(0).array().inverse();
            temp_explore = (*par).weights[i_kernel]->col(0)-(*current_param).weights[i_kernel]->col(0);
            param_nom = param_nom.array() + temp_W.array()*temp_explore.array()*reward_j;
            param_dnom = param_dnom.array() + temp_W.array()*reward_j;
        }

        MatrixXf *new_weights = new MatrixXf(n_weights,1);
        *new_weights = (*current_param).weights[i_kernel]->col(0).array() + param_nom.array()/(param_dnom.array()+0.0000000001f);

        Eigen::VectorXf var_nom = VectorXf::Zero(n_weights).transpose();
        float var_dnom = 0;
        n_items = (*episode-1<(*config).elite_variance)?*episode-1:(*config).elite_variance;

        for(int i=0;i<n_items;i++){
            float reward_j = best_params[best_params.size()-1-i].first;
            Param* par = best_params[best_params.size()-1-i].second;
            temp_explore = (*par).weights[i_kernel]->col(0)-(*current_param).weights[i_kernel]->col(0);
            var_nom = var_nom.array() + temp_explore.array().square()*reward_j;
            var_dnom = var_dnom + reward_j;
        }

        MatrixXf matr_var=MatrixXf::Zero(n_weights,1);
        matr_var.col(0) = var_nom.array() / (var_dnom);

        /*
        int best = s_Return[s_Return.size()-1].second;
        float mean = param.col(best).mean();
        float var_iance = ((param.col(best).squaredNorm()/param.col(best).rows())-pow(mean,2))*100/n_motors;
        VectorXf varVector = VectorXf::Constant(n_motors,var_iance).array().transpose();
        cout << var_iance << endl;
        variance.col(iter)=variance.col(iter).cwiseMin(10*varVector);
        if(var_iance>0.1)
        variance.col(iter)=variance.col(iter).cwiseMax(1e-20*varVector);
        else
        */

        float coef = pow((*config).d_variance,*episode);
        if(coef<1e-15)
            coef = 1e-15;

        matr_var.col(0)=matr_var.col(0).cwiseMin(coef*10.f*variances[i_kernel].col(0));
        matr_var.col(0)=matr_var.col(0).cwiseMax(coef*0.1f*variances[i_kernel].col(0));

        MatrixXf *new_variance = new MatrixXf(matr_var);

        if(*episode<3)
            *new_variance = variances[i_kernel].col(0);

        (*new_param).variance[i_kernel]=new_variance;
        Eigen::VectorXf noise = normalDistribution(n_weights);
        (*new_weights)=(*new_weights).col(0).array() + (*new_variance).col(0).array().sqrt()*noise.array();
        (*new_param).weights[i_kernel]=new_weights;

        } else {
            /*Eigen::VectorXf noise = normalDistribution(n_weights);
            MatrixXf *weights = new MatrixXf(MatrixXf::Zero(n_weights,1));
            MatrixXf *variance = new MatrixXf(variances[i_kernel].col(*iter));
            *weights=variances[i_kernel].col(*iter).array().sqrt()*noise.array();
            (*new_param).weights.push_back(weights);
            (*new_param).variance.push_back(variance);*/
        }

    }

    if(*episode>1)
        current_param_episode = new_param;

    for(unsigned int i_kernel=0;i_kernel<(*config).n_motors;i_kernel++){
        //if(((((*iter/100+i_kernel))%(n_motors))==0)||*iter<2){
        if(*episode>1)
        kernels[i_kernel].setWeights((*current_param_episode).weights[i_kernel]->col(0));
        else
        kernels[i_kernel].setWeights((*current_param).weights[i_kernel]->col(0));
        //  cout << "maj " << *iter << " " << i_kernel << endl;
        //}
    }
}

Eigen::VectorXf Algo::normalDistribution(unsigned int _size){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0,1.0);
    //distribution.reset();
    Eigen::VectorXf retour(_size);
    for(unsigned int i=0;i<_size;i++)
        retour(i) = distribution(generator);

    return retour;
}

vector<float> Algo::getNextActions(vector<float> sensors){
    vector<float> actions;
    for (unsigned int i=0;i<(*config).n_motors;i++){
          vector<float> states;
          //float coef = 1.0f/0.75f;
          float coef = 1.0f;
          float theta(0);
          float y(0);
          float x(0);

        if(i==0){
          theta +=sensors[i*2];
          y+= -cos(theta);
          x+= sin(theta);
          //states.push_back(x/1.f);
          //states.push_back(y/1.f);
          states.push_back(sensors[0]/PI);
          states.push_back(coef*sensors[1]/27.f);
          states.push_back(sensors[2]/PI);
          states.push_back(coef*(sensors[3])/62.f);
          //states.push_back(sensors[0]+sensors[2]);
          //states.push_back(coef*(sensors[1]+sensors[3])/36.f);
        } else if(i==1) {
          theta +=sensors[i*2];
          y+= -cos(theta);
          x+= sin(theta);
          //states.push_back(x/2.f);
          //states.push_back(y/2.f);
          //states.push_back(sensors[0]);
          //states.push_back(coef*sensors[1]/28.f);
          states.push_back((sensors[0]+sensors[2])/PI);
          states.push_back(coef*(sensors[1]+sensors[3])/45.f);
          states.push_back(sensors[4]/PI);
          states.push_back(coef*sensors[5]/71.f);
          //states.push_back(sensors[2]+sensors[4]);
          //states.push_back(coef*(sensors[3]+sensors[5])/44.f);
            /*
          states.push_back(sensors[0]+sensors[2]);
          states.push_back(coef*(sensors[1]+sensors[3])/36);
          states.push_back(sensors[4]);
          states.push_back(coef*sensors[5]/70);
          states.push_back(sensors[6]);
          states.push_back(coef*sensors[7]/80);
          */
        } else if(i==2) {
          theta +=sensors[i*2];
          y+= -cos(theta);
          x+= sin(theta);
          //states.push_back(x/3.f);
          //states.push_back(y/3.f);
          //states.push_back(sensors[0]+sensors[2]);
          //states.push_back((sensors[1]+sensors[3])/27);
          states.push_back((sensors[0]+sensors[2]+sensors[4])/PI);
          states.push_back(coef*(sensors[1]+sensors[3]+sensors[5])/44.f);
          states.push_back(sensors[6]/PI);
          states.push_back(coef*sensors[7]/95.f);
          //states.push_back(sensors[2]+sensors[4]+sensors[6]);
          //states.push_back(coef*(sensors[3]+sensors[5]+sensors[7])/41.f);
        } else if(i==3) {
          theta +=sensors[i*2];
          y+= -cos(theta);
          x+= sin(theta);
          //states.push_back(x/4.f);
          //states.push_back(y/4.f);
          //states.push_back(sensors[0]+sensors[2]);
          //states.push_back((sensors[1]+sensors[3])/27);
          states.push_back((sensors[0]+sensors[2]+sensors[4]+sensors[6])/PI);
          states.push_back(coef*(sensors[1]+sensors[3]+sensors[5]+sensors[7])/41.f);
          states.push_back(sensors[8]/PI);
          states.push_back(coef*sensors[9]/85.f);
        }
        actions.push_back(kernels[i].getValue(states,4));
    }
return actions;
}
   void Algo::save(const std::string& path) {

        std::cout.setf(std::ios::scientific);
       ofstream fichier(path, ios::out | ios::trunc);  // ouverture en écriture avec effacement du fichier ouvert
        if(fichier)
        {
            Param *p = best_params[best_params.size()-1].second;

            for(unsigned int i=0;i<(*p).weights.size();i++){
                 for (int k=0;k<(*p).weights[i]->col(0).size();k++){
                    fichier << Kernel::GetBinary32((*((*p).weights[i]))(k,0)) << endl;
                }
            }
            /*
            for(unsigned int i=0;i<n_weights;i++){
                for (unsigned int k=0;k<n_motors;k++){
                    //fichier << scientific << std::setprecision(17) << params[k](i,j) << endl;
                    fichier << Kernel::GetBinary32(params[k](i,j)) << endl;
                }
            }
*/
          fichier.close();
        }
        else
                cerr << "Impossible d'ouvrir le fichier !" << endl;


  }

  void Algo::load(const std::string& path) {

        ifstream fichier(path, ios::in);  // on ouvre en lecture
        if(fichier)
        {
            string ligne;
            unsigned int i(0);
            int i_kernel(0);
            MatrixXf *weights = new MatrixXf(MatrixXf::Zero(n_weights,1));
            //*weights=variances[i_kernel].col(*iter).array().sqrt()*noise.array();

            while(getline(fichier, ligne))
            {

                float temp;
                temp = Kernel::GetFloat32(ligne.c_str());
                //temp = ::atof(ligne.c_str());

                //cout << i << "  " << i/n_motors << "  " << i%n_motors << endl;
                //cout << i << " " << temp << endl;

                (*weights)(i,0)=temp;
                //params[i_motor](i_kernel,0)=temp;
                i++;
                if(i>=kernels[i_kernel].getSize()){

                    (*current_param).weights[i_kernel]=weights;
                    /*if(i_kernel==1 && n_motors==3){
                         (*current_param).weights[i_kernel+1]=weights;
                    break;
                    }*/
                    i_kernel++;
                    //if(i_kernel==n_motors)
                    //    break;
                    i=0;
                    weights = new MatrixXf(MatrixXf::Zero(n_weights,1));
                }
            }

                fichier.close();
            for (unsigned int k=0;k<(*config).n_motors;k++){
               // params[k].col(1)=params[k].col(0);
            }
        }
        else
                cerr << "Impossible d'ouvrir le fichier !" << endl;

  }
