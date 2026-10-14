#include <Kin/kin.h>

struct StableConfigSampler_Options {
  RAI_PARAM("StableConfigSampler/", int, verbose, 1)
  RAI_PARAM("StableConfigSampler/", double, frictionCone_mu, .8)
  RAI_PARAM("StableConfigSampler/", bool, resolveInitCollisions, false)
};

struct SampleOutput;

struct StableConfigSampler{
  rai::Configuration C;
  StableConfigSampler_Options opt;
  str config_file;
  strA must_supp_env, supp_env, supp_bot, collides;
  str com_name;
  uint min_n_supports, max_n_supports;
  rai::Graph config_dict;
  rai::Frame* com;
  double total_mass;
  strA com_frames;

  str data_file;
  bool savePngs=false;

  StableConfigSampler(str config_file, strA must_supp_env, strA supp_env, strA supp_bot, str com_name,
                     strA collides, uint min_n_supports, uint max_n_supports);

  SampleOutput sample(rai::Configuration& C) const;

  void run();

  void run_parallelized(uint n_threads=10);

  void write_h5(double time, const SampleOutput& data);
};
