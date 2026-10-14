#include <DataGen/rndStableConfigs.h>
#include <Kin/viewer.h>
#include <Kin/frame.h>
#include <KOMO/komo.h>
#include <Kin/dof_forceExchange.h>
#include <Optim/NLP_Solver.h>
#include <Kin/F_forces.h>
#include <Optim/NLP_Sampler.h>

#include <Core/h5.h>

//===========================================================================

void rndStableConfigs(str config_file, strA mustSupp, strA rndSupp, strA _nil1, str _nil2, strA collides, uint min_n_supports, uint max_n_supports){
  // rnd.seed_random();

  rai::Configuration C;
  C.addFile(config_file);
  rai::Graph config_dict = C.asDict();

  RndStableConfigs RC;

  //modifications
  for(rai::Frame* f : C.getJoints()) if(!f->joint->sampleUniform) f->joint->sampleSdv = .4;
  for(rai::Frame* f : C.frames) if(f->shape) if(!f->shape->cont && f->shape->_type==rai::ST_mesh) f->setContact(-2);
  // FrameL F;
  // for(rai::Frame* f : C.frames) if(f->name.contains("panda_coll") || f->name.contains("palm")) F.append(f);
  // for(rai::Frame* f : F) delete f;

  if(RC.opt.verbose>0) C.view(false);
  C.makeMeshesSSCvx(.002);
  // double c = C.coll_totalViolation();
  // if(c>0.){
  //   C.coll_reportProxies();
  //   HALT("need a collision free scene!");
  // }
  if(RC.opt.verbose>0) C.view(false);
  rai::wait(.5);
  //  C.view(true); return;

  // for(uint k=0;k<10;k++){  C.setRandom(0, 10);  C.view(true);  }


  //C.view();
  //C.get_viewer()->setWindow("stable configs", 800, 600);
  //RC.savePngs = true;

  uint n=rai::getParameter<int>("num_samples", 100);
  arr positions(n, C.getJointStateDimension());
  StringAA supports (n);
  intAA pairs(n);
  arrA forces(n);

  double time = -rai::cpuTime();
  for(uint i=0;i<n;){
    bool succ = RC.getSample(C, mustSupp, rndSupp, collides, max_n_supports, min_n_supports);
    if(succ){
      positions[i] = C.getJointState();
      supports(i) = RC.supp;
      pairs(i) = RC.pairs.reshape(-1);
      forces(i) = RC.forces.reshape(-1);
      i++;
    }
    double t = rai::cpuTime()+time;
    cout <<"\r--- samples: " <<i <<'/' <<n <<" ETA: " <<rai::niceTime(t/i * (n-i)) <<std::flush;
  }
  time += rai::cpuTime();
  cout <<"=== sampling report ===" <<endl;
  RC.report();
  cout <<"  feasible rate: " <<double(RC.totalSucc)/RC.totalRuns
       <<"  time/sample: " <<1000.*time/RC.totalSucc <<"msec"
       <<"  queries/sample: " <<double(RC.totalEvals)/RC.totalSucc
       <<"  time/query: " <<1000.*time/RC.totalEvals <<"msec"
       <<"  total time: " <<time <<"sec" <<endl;

  str configs_file = STRING(config_file.sub(0,-4) <<".h5");

  cout <<"=== writing data: " <<configs_file <<" ===" <<endl;

  rai::H5_Writer h5(configs_file);
  auto manifest = rai::Graph({{"info", "random stable configurations data"},
                              {"original config file", config_file},
                              {"free_frames", StringA({"obj"})},
                              {"potential_supports", (mustSupp, rndSupp)},
			      {"time", time},
			      {"evals", RC.totalEvals},
			      {"runs", RC.totalRuns}});
  h5.writeDict("manifest", manifest);
  h5.writeDict("params", rai::params());
  h5.writeDict("config", config_dict);
  h5.write("q", positions);
  h5.writeA("contacts", pairs);
  h5.writeA("forces", forces);

}

//===========================================================================

#include "StableConfigSampler.h"

//===========================================================================

void rndStableConfigs_new(str config_file, strA must_supp_env, strA supp_env, strA supp_bot, str com_name, strA collides, uint min_n_supports, uint max_n_supports){
  uint n_threads = rai::getParameter<uint>("num_threads");
  cout <<"=== Stable Config Sampler - " <<config_file <<" (n_threads: " <<n_threads <<") ===" <<endl;
  StableConfigSampler CS(config_file, must_supp_env, supp_env, supp_bot, com_name, collides, min_n_supports, max_n_supports);
  if(!n_threads){
    CS.run();
  }else{
    CS.run_parallelized(n_threads);
  }
}

int main(int argc,char** argv){
  rai::initCmdLine(argc,argv);

  str scene = rai::getParameter<str>("scene");

  if(scene=="double_sphere"){
    rndStableConfigs_new("../../scenes/double_sphere.yml",
		     {},
		     { "table", "box", "wall", "fing" },
		     { "obj", "obj", "obj", "obj" },
		      "obj",
		     {},
		     1, 3);

  }else if(scene=="panda_sphere_all"){
    rndStableConfigs("../../scenes/panda_sphere.yml",
		     {},
		     { "panda_finger_joint1",  "panda_rightfinger_0",  "panda_hand_joint",
			 "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
			 "panda_link0",
			 "table",
			 "box", "wall"
		     },
		     {}, "obj",
		     {},
		     1, 3);

  }else if(scene=="panda_sphere"){
    rndStableConfigs_new("../../scenes/panda_sphere.yml",
		     {},
		     { "panda_finger_joint1", "panda_rightfinger_0", "panda_hand_joint",
		      "panda_link0", "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
		      "table",
		      // "box", "wall",
		      },
		     {"obj","obj","obj","obj","obj","obj","obj","obj","obj","obj","obj","obj"},
		     "obj",
		     {},
		     1, 3);

  }else if(scene=="sphere_cube"){
    rndStableConfigs_new("../../scenes/sphere_cube.yml",
                     { "table" },
                     { "fing" }, //, "box", "wall", "fing" },
                     {"obj","obj"},
                     "obj",
                     {},
                     1, 2);

  }else if(scene=="panda_cube"){
    rndStableConfigs_new("../../scenes/panda_cube.yml",
                     { "table" },
                     { "panda_finger_joint1", "panda_rightfinger_0", "panda_hand_joint",
                      // "table",
                      // "box", "wall",
                      // "panda_link0", "panda_joint1",
                      //"panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
                      },
                     {"obj","obj","obj","obj"},
                     "obj",
                     {}, // "box", "wall" },
                     2, 3);

  }else if(scene=="allegro_cube"){
    rndStableConfigs_new("../../scenes/allegro_cube.yml",
                         {},
                         { "tip1", "tip2", "tip3", "tip4", },
                         { "obj", "obj", "obj", "obj" },
                         "obj",
                         {},
                         2, 4);

  }else if(scene=="allegroDown_cube"){
    rndStableConfigs_new("../../scenes/allegroDown_cube.yml",
                     {},
                     { "tip1", "tip2", "tip3", "tip4", },
                     { "obj", "obj", "obj", "obj" },
                     "obj",
                     {},
                     2, 4);

  }else if(scene=="g1_floor"){
    rndStableConfigs_new("../../scenes/g1_floor.yml",
                          {},
                          { "table", "table" },
                          { "foot_l", "foot_r" },
                          "g1_com",
                          {"right_shoulder_yaw_joint", "waist_pitch_joint", "left_shoulder_yaw_joint", "waist_pitch_joint"},
                          1, 2);

  }else if(scene=="sphere_T"){
    rndStableConfigs_new("../../scenes/sphere_T.yml",
                         {},
                         { "table", "fing" },
                         { "obj_T1", "obj_T2" },
                         "obj",
                         { "obj_T1", "table", "obj_T2", "table", "fing", "table" },
                         1, 3);

  }else if(scene=="pandaS_cube"){
    rndStableConfigs("../../scenes/pandaS_cube.yml",
                     { "table" },
                     { "fing" },
                     {}, "obj",
                     { "panda_finger_joint1", "panda_rightfinger_0", "panda_hand_joint" }, //collides
                     2, 1);

  }else if(scene=="4sphere_cube"){
    rndStableConfigs("../../scenes/4sphere_cube.yml",
                     {},
                     { "1_fing", "2_fing", "3_fing", "4_fing" },
                     {}, "obj",
                     {},
                     4, 2);
  }else if(scene=="pandas_cube"){
    rndStableConfigs("../../scenes/pandas_cube.yml",
		     {},
		     { "l_panda_finger_joint1", "l_panda_rightfinger_0",
			 "l_panda_hand_joint",
			 "r_panda_finger_joint1", "r_panda_rightfinger_0",
			 "r_panda_hand_joint"
		     },
		     {}, "obj",
		     {},
		     3, 1);
  }else if(scene=="tesollo"){
    rndStableConfigs("../../scenes/tesollo_cube.yml",
		     {},
		     { "rl_dg_2_4_10_shape2", "rl_dg_3_4_14_shape2", "rl_dg_4_4_18_shape2", "rl_dg_5_4_22_shape2", "rl_dg_1_4_6_shape2"
			 // "l_panda_finger_joint1", "l_panda_rightfinger_0", //"l_panda_hand_joint",
			 // "r_panda_finger_joint1", "r_panda_rightfinger_0", //"r_panda_hand_joint"
		     },
		     {}, "obj",
		     {},
		     3, 1);
  }else NIY;
  // rndStableConfigs("../../scenes/twoFingersBall.yml", { "table", "box", "wall", "l_fing", "r_fing" });
  // rndStableConfigs("../../scenes/twoFingersCube.yml", { "table", "box", "wall", "l_fing", "r_fing" });

  return 0;
}

