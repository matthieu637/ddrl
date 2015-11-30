#ifndef CONFIG_HPP
#define CONFIG_HPP
  struct Config {
      unsigned int n_episodes;
      unsigned int n_instances;
      unsigned int n_steps_max;
      unsigned int n_motors;
      unsigned int n_sensors;
      unsigned int n_states_per_kernels;
      unsigned int n_basis_per_dim;
      unsigned int elite;
      unsigned int elite_variance;
      double width_kernel;
      double d_variance;
      double var_init;
      //std::vector<unsigned int> n_kernels_per_dim;
    };
#endif // CONFIG_HPP
