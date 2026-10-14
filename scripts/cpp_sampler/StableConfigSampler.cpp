#include "StableConfigSampler.h"

#include <Core/h5.h>
#include <Core/thread.h>
#include <Kin/frame.h>
#include <Kin/viewer.h>
#include <Kin/F_forces.h>
#include <KOMO/komo.h>
#include <Optim/NLP_Solver.h>

StableConfigSampler::StableConfigSampler(str config_file, strA must_supp_env, strA supp_env, strA supp_bot, str com_name, strA collides, uint min_n_supports, uint max_n_supports)
    : config_file(config_file), must_supp_env(must_supp_env), supp_env(supp_env), supp_bot(supp_bot), collides(collides), com_name(com_name), min_n_supports(min_n_supports), max_n_supports(max_n_supports){
  C.addFile(config_file);
  config_dict = C.asDict();
  data_file = STRING(config_file.sub(0,-4) <<".h5");

  //-- config modifications
  for(rai::Frame* f : C.getJoints()) if(!f->joint->sampleUniform) f->joint->sampleSdv = .4;
  for(rai::Frame* f : C.frames) if(f->shape) if(!f->shape->cont && f->shape->_type==rai::ST_mesh) f->setContact(-3);

  //-- check collisions
  if(opt.verbose>1) C.view(false);
  C.makeMeshesSSCvx(.002);
  double c = C.coll_totalViolation();
  if(c>0.){
    C.coll_reportProxies();
    HALT("need a collision free scene!");
  }
  if(opt.verbose>1) C.view(false);

  //-- collect robot inertial frames
  com = C.getFrame(com_name);
  if(com->joint && com->joint->type==rai::JT_trans3){ //that's a floating 3D com frame
    CHECK(!com->inertia, "");
    com_frames = {com_name};
    FrameL sub = com->parent->getSubtree();
    total_mass=0.;
    for(auto* f:sub) if(f->inertia && f!=com){ com_frames.append(f->name); total_mass += f->inertia->mass; }
    // total_mass = 1.;
    cout <<"COM frames: " <<com_frames <<endl;
    cout <<"total mass: " <<total_mass <<endl;
  }else{
    com_frames = {};
    total_mass=com->inertia->mass;
    cout <<"COM frames: " <<com_frames <<endl;
    cout <<"total mass: " <<total_mass <<endl;
  }
}

struct SampleOutput{
  arr positions;
  intAA contacts;
  intAA pairs;
  arrA forces;
  uint evals=0, runs=0, succ=0;
  bool isFeasible=true;
  SampleOutput& failed(uint _evals){ evals=_evals; isFeasible=false; return *this; }
  void append(SampleOutput& s){
    evals+=s.evals;
    runs+=1;
    if(s.isFeasible){
      succ+=1;
      positions.append(s.positions).reshape(succ, -1);
      contacts.append(s.contacts);
      pairs.append(s.pairs);
      forces.append(s.forces);
    }
  }
  void report(double time) const{
    cout <<"=== sampling report ===" <<endl;
    cout <<"  feasible samples: " <<succ
         <<"  feasible rate: " <<double(succ)/runs
         <<"  time/sample: " <<1000.*time/succ <<"msec"
         <<"  queries/sample: " <<double(evals)/succ
         <<"  time/query: " <<1000.*time/evals <<"msec"
         <<"  total time: " <<time <<"sec" <<endl;
  }
};

SampleOutput StableConfigSampler::sample(rai::Configuration& C) const{
  C.setRandom(0, 0);

  arr qRnd = C.getJointState();

  if(opt.resolveInitCollisions){ //first make configuration collision free
    KOMO komo(C, 1,1,0, true);
    komo.addObjective({}, FS_qItself, {}, OT_sos, {1e-2}, qRnd);
    komo.addObjective({}, FS_accumulatedCollisions, {}, OT_eq, {1e1} );
    for(uint i=0;i<collides.N;i+=2){
      komo.addObjective({}, FS_negDistance, {collides(i), collides(i+1)}, OT_ineq, {1e1});
    }
    komo.addQuaternionNorms();

    if(opt.verbose>4) komo.opt.animateOptimization = opt.verbose-4;
    if(opt.verbose>2){
      komo.view(true, "before collide resolution");
    }
    auto ret = rai::NLP_Solver(komo.nlp(), opt.verbose)
                   .solve();
    C.setJointState(ret->x);
    if(opt.verbose>2){
      cout <<*ret <<endl;
      komo.view(true, STRING("after collide resolution\n" <<*ret));
      // C.view(true, "no collide");
    }
    if(!ret->feasible) return SampleOutput().failed(ret->evals);
  }

  C.coll_engine(); //ensure the given C has coll_engine, which is referenced by komo, instead of recreating a new one each time
  KOMO komo(C, 1,1,0, true);
  komo.addObjective({}, FS_qItself, {}, OT_sos, {1e-2}, qRnd);
  komo.addObjective({}, FS_jointLimits, {}, OT_ineq, {1e0});
  komo.addObjective({}, FS_accumulatedCollisions, {}, OT_eq, {1e1} );
  for(uint i=0;i<collides.N;i+=2){
    komo.addObjective({}, FS_negDistance, {collides(i), collides(i+1)}, OT_ineq, {1e2});
  }

  komo.addObjective({}, make_shared<F_TotalForce>(total_mass * arr{0., 0., 9.81}), {com_name}, OT_eq, {1e1/total_mass} );
  if(com_frames.N){
    komo.addObjective({}, make_shared<F_EqCenterOfMass>(), com_frames, OT_eq, {1e1} );
  }

  //-- discrete decisions:
  CHECK_GE(max_n_supports, min_n_supports, "");
  uint n_supports = rnd.uni_int(min_n_supports, max_n_supports);
  uintA perm_env = rai::randperm(supp_env.N);
  uintA perm_bot = rai::randperm(supp_bot.N);
#if 0
  strA supp0, supp1;
  for(uint i=0;i<n_supports;i++) supp0.append(supp_bot(perm_bot(i)));
  if(must_supp_env.N) supp1.append(must_supp_env.rndElem());
  while(supp1.N<n_supports) supp1.append(supp_env(perm_env.popFirst()));
#else
  //make a matrix
  boolA contact(must_supp_env.N+supp_env.N, supp_bot.N);
  contact = false;
  uint n=0;
  CHECK(!must_supp_env.N, "");
  // for(uint i=0;i<must_supp_env.N;i++){ contact(i, supp_bot.rndElem()) = true; n++; }
  while(n<n_supports){
    unsigned char& c=contact.rndElem();
    if(!c){ c=true; n++; }
  }
  strA supp0, supp1;
  for(uint i=0;i<contact.d0;i++) for(uint j=0;j<contact.d1;j++){
      if(contact(i,j)){ supp1.append(supp_env(i)); supp0.append(supp_bot(j)); }
    }
#endif

  if(opt.verbose>0){
    LOG(0) <<"\n======================\n" <<supp0 <<supp1;
  }

  //-- contact dofs and constraints
  for(uint i=0;i<n_supports;i++){
    komo.addContactForceFrame({1.}, supp0(i), supp1(i), opt.frictionCone_mu, total_mass / double(n_supports), com_name);
  }

  //always last:
  komo.addQuaternionNorms();

  if(opt.verbose>0) komo.set_viewer(C.get_viewer());
  if(opt.verbose>2){
    if(opt.verbose>3){
      cout <<komo.report(true) <<endl;
    }
    komo.view(true, STRING("INIT\n supports: " <<supp0 <<supp1));
  }

  if(opt.verbose>4) komo.opt.animateOptimization = opt.verbose-4;
#if 1
  auto ret = rai::NLP_Solver(komo.nlp(), opt.verbose-1)
                 // .setInitialization(komo.pathConfig.getJointState())
                 .solve();
#else
  auto ret = NLP_Sampler(komo.nlp())
                 // .setOptions(NLP_Sampler_Options().set_slackMaxStep(.2). set_downhillMaxSteps(100). set_slackRegLambda(.1) .set_tolerance(1e-3) .set_seedMethod("gauss"))
                 .sample();
#endif

  ret->setFeasible(1e-2);

  intA pairs;
  arr forces;
  std::tie(pairs,forces) = komo.pathConfig.getForceArrays();

  intA contacts;
  for(uint i=0;i<n_supports;i++){ contacts.append(C.getFrame(supp0(i))->ID); contacts.append(C.getFrame(supp1(i))->ID); }

  if(opt.verbose>0){
    cout <<"\nkomo (setup: " <<komo.timeSetup <<" setJoints:" <<komo.timeSetJoints <<" broadphase:" <<komo.timeBroadphase <<" features:" <<komo.timeFeatures <<")" <<endl;
    LOG(0) <<*ret;
  }
  if(opt.verbose>2){
    cout <<"FORCES:\n" <<pairs <<forces <<endl;
  }

  if(!ret->feasible){
    if(opt.verbose>2) komo.view(opt.verbose>3, STRING("FAIL\nsupports: " <<supp0 <<supp1 <<"\n" <<*ret));
    return SampleOutput().failed(ret->evals); //FAILURE RETURN
  }

  if(opt.verbose>0) komo.view(opt.verbose>1, STRING("SUCCESS\nsupports: " <<supp0 <<supp1 <<"\n" <<*ret));
  if(savePngs) C.get_viewer()->savePng();

  arr q = komo.getPath_qOrg()[0].copy();
  C.setJointState(q);
  return SampleOutput{ q, {contacts}, {pairs.reshape(-1)}, {forces.reshape(-1)} };
}

void StableConfigSampler::write_h5(double time, const SampleOutput& data){
  cout <<"=== writing data: " <<data_file <<" ===" <<endl;
  rai::H5_Writer h5(data_file);
  auto manifest = rai::Graph({{"info", "random stable configurations data"},
                              {"original config file", config_file},
			      {"frame_names", C.getFrameNames()},
                              {"supports_env", supp_env},
                              {"supports_bot", supp_bot},
                              {"com_name", com_name},
                              {"total_mass", total_mass},
                              {"time", time},
                              {"evals", data.evals},
                              {"runs", data.runs},
                              {"succ", data.succ}});
  h5.writeDict("manifest", manifest);
  h5.writeDict("params", rai::params());
  h5.writeDict("config", config_dict);
  h5.write("q", data.positions);
  h5.writeA("contacts", data.contacts);
  h5.writeA("force_pairs", data.pairs);
  h5.writeA("forces", data.forces);
}

void StableConfigSampler::run(){
  uint n=rai::getParameter<int>("num_samples", 100);
  SampleOutput data;

  C.coll_engine();

  double time = -rai::realTime();
  for(uint i=0;;i++){
    SampleOutput s = sample(this->C);
    data.append(s);
    double t = rai::realTime()+time;
    cout <<"\r--- samples: " <<data.succ <<'/' <<n <<"  time/sample: " <<1000.*t/data.succ <<"msec"<< "  ETA: " <<rai::niceTime(t/data.succ * (n-data.succ)) <<std::flush;
    if(data.succ>=n) break;
  }
  time += rai::realTime();

  data.report(time);

  write_h5(time, data);
}

void StableConfigSampler::run_parallelized(uint n_threads){
  uint n=rai::getParameter<int>("num_samples", 100);
  rai::Var<SampleOutput> data;

  struct Worker : rai::Thread {
    rai::Var<SampleOutput>& data;
    StableConfigSampler &SCS;
    rai::Configuration Ccopy;
    Worker(rai::Var<SampleOutput>& data, StableConfigSampler &SCS) : Thread(""), data(data), SCS(SCS){
      Ccopy.addConfigurationCopy(SCS.C);
      Ccopy.coll_engine();
    }
    ~Worker(){ threadClose(); }
    void step(){
      auto s = SCS.sample(Ccopy);
      data.set()->append(s);
    }
  };

  double time = 0.;
  {
    rai::Array<shared_ptr<rai::Thread>> threads;
    LOG(0) <<"creating " <<n_threads <<" threads ...";
    for(uint k=0;k<n_threads;k++) threads.append(make_shared<Worker>(data, *this));
    LOG(0) <<"starting threads ...";
    time = rai::realTime();
    for(uint k=0;k<n_threads;k++) threads(k)->threadLoop();
    for(;;){
      rai::wait(.5);
      uint nd=data.get()->succ;
      if(nd>=n) break;
      double t = rai::realTime()-time;
      cout <<"\r--- samples: " <<nd <<'/' <<n <<"  time/sample: " <<1000.*t/nd <<"msec"<< "  ETA: " <<rai::niceTime(t/nd * (n-nd)) <<std::flush;
    }
  }
  time = rai::realTime()-time;
  cout <<endl;

  data.get()->report(time);

  write_h5(time, data.get());
}
