import time
from os import path

import numpy as np
import omegaconf
import robotic as ry
from robotic.tools import h5_helper

from csrl.envs.helpers import process_cfg


class StableConfigSampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.totalRuns = 0
        self.totalEvals = 0
        self.totalSucc = 0

        self.C = ry.Config()
        self.C.addFile(self.cfg.scene_yaml)
        self.config_dict = self.C.asDict()

        assert self.C.getFrame("obj") is not None
        for s in self.cfg.supports_env:
            assert self.C.getFrame(s) is not None
        for s in self.cfg.supports_float:
            assert self.C.getFrame(s) is not None

        # MISSING in python!
        # for(rai::Frame* f : C.getJoints()) if(!f->joint->sampleUniform) f->joint->sampleSdv = .4;
        # for(rai::Frame* f : C.frames) if(f->shape) if(!f->shape->cont && f->shape->_type==rai::ST_mesh) f->setContact(-2);

        ## C++ checks collisions in initial scene

        com = self.C.getFrame(self.cfg.com_name)
        if com.getJointType == ry.JT.trans3:  # that's a floating 3D com frame
            assert com.getMass() == 0, ""
            # raise NotImplementedError()
            self.com_frames = [com_name]
            sub = com.getParent().getSubtree()
            self.total_mass = 0.0
            for f in sub:
                if f.getMass() > 0.0:
                    self.com_frames.append(f.name)
                    self.total_mass += f.getMass()
        else:
            self.com_frames = []
            self.total_mass = com.getMass()
        print("COM frames: ", self.com_frames)
        print("total mass: ", self.total_mass)

    def sample(self, C: ry.Config):
        C.setRandom()

        qRnd = C.getJointState()

        ## C++ code here has solver to first get collision free config (esp for hands!)

        C.computeCollisions()  # ensure the given C has coll_engine, which is referenced by komo, instead of recreating a new one each time
        komo = ry.KOMO(C, 1, 1, 0, True)
        komo.addObjective([], ry.FS.qItself, [], ry.OT.sos, [1e-2], qRnd)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])

        ## C++ code allows explicit collides
        # for(uint i=0;i<collides.N;i+=2){
        # komo.addObjective({}, FS_negDistance, {collides(i), collides(i+1)}, OT_ineq, {1e1});

        ## C++ code sets proper mass
        komo.addObjective([], ry.FS.totalForce, ["obj"], ry.OT.eq, [1e1])
        ## C++ works for floating
        # if(com_frames.N){
        # komo.addObjective({}, make_shared<F_EqCenterOfMass>(), com_frames, OT_eq, {1e1} );

        ##-- discrete decisions:

        n_supports = np.random.randint(self.cfg.min_n_supports, self.cfg.max_n_supports + 1)
        assert n_supports >= self.cfg.min_n_supports and n_supports <= self.cfg.max_n_supports
        n_env, n_float = len(self.cfg.supports_env), len(self.cfg.supports_float)
        # perm_env = np.random.permutation(n_env)
        # perm_float = np.random.permutation(n_float)
        # make a matrix
        contact = np.zeros(n_env * n_float, dtype=np.bool)
        indices = np.random.choice(contact.size, size=n_supports, replace=False)
        contact[indices] = True
        contact = contact.reshape(n_env, n_float)
        indices = np.where(contact)
        print(indices)
        supp_env = [self.cfg.supports_env[int(i)] for i in indices[0]]
        supp_float = [self.cfg.supports_float[int(i)] for i in indices[1]]

        if self.cfg.verbose > 0:
            print("\n======================\n", supp_env, supp_float)

        ## -- contact dofs and constraints
        for i in range(n_supports):
            komo.addContactForceFrame(
                [1.0],
                supp_float[i],
                supp_env[i],
                self.cfg.frictionCone_mu,
                init_objMass=self.total_mass / n_supports,
                obj_com_frame="obj",
            )

        ## always last
        komo.addQuaternionNorms()

        if self.cfg.verbose > 0:
            komo.set_viewer(C.get_viewer())

        ret = ry.NLP_Solver(komo.nlp(), self.cfg.verbose - 1).solve()

        tol = 1e-2
        ret.feasible = (ret.ineq < tol) and (ret.eq < tol)

        pairs, forces = komo.getConfig().getForceArrays()

        contacts = [[C.getFrame(supp_env[i]).ID, C.getFrame(supp_float[i]).ID] for i in range(n_supports)]
        contacts = np.array([c for pair in contacts for c in pair], dtype=np.int32)

        if self.cfg.verbose > 0:
            print(ret)
            if self.cfg.verbose > 1:
                print(pairs, forces)

        self.totalRuns += 1
        if not ret.feasible:
            if self.cfg.verbose > 2:
                komo.view(self.cfg.verbose > 3, f"failed: {ret}")
        else:
            self.totalSucc += 1
            self.totalEvals += ret.evals
            if self.cfg.verbose > 0:
                komo.view(self.cfg.verbose > 1, f"supp {ret}")
            if self.cfg.savePngs:
                C.get_viewer().savePng()

            q = komo.getPath()[0]
            return True, q, contacts, pairs, forces

        return False, None, None, None, None

    def run(self):
        self.C.view(False)
        self.C.makeMeshesSSCvx(radius=0.002)
        c = self.C.coll_totalViolation()
        if c > 0.0:
            self.C.coll_reportProxies()
            raise Exception("need a collision free scene!")

        self.C.view(False)
        self.C.coll_totalViolation()  # awkward: this triggers creation of collision engin, which is passed to KOMO - more efficient
        time.sleep(0.5)

        q = []
        contacts = []
        pairs = []
        forces = []

        run_time = -time.time()

        num_samples = 0
        while num_samples < self.cfg.num_samples:
            succ, _q, _conts, _pairs, _forces = self.sample(self.C)
            if succ:
                q.append(_q)
                contacts.append(_conts)
                pairs.append(_pairs)
                forces.append(_forces)
                num_samples += 1
            print(f"\r-- samples: {num_samples}", end="")

        run_time += time.time()

        print()
        print("=== sampling report ===")
        print(f"time/sample: {1000.0 * run_time / self.totalSucc}msec   time/query: {1000.0 * run_time / self.totalEvals}msec  total time: {run_time}sec")

        filename = path.splitext(path.basename(self.cfg.scene_yaml))[0]
        output_file = f"{filename}_configs.h5"

        print("=== writing data: ", output_file, " ===")

        h5 = h5_helper.H5Writer(output_file)
        manifest = {
            "info": "random stable configurations data",
            "original config file": self.cfg.scene_yaml,
            "frame_names": self.C.getFrameNames(),
            "supports_env": self.cfg.supports_env,
            "supports_bot": self.cfg.supports_float,
            "com_name": self.cfg.com_name,
            "total_mass": self.total_mass,
            "time": run_time,
            "evals": self.totalEvals,
            "runs": self.totalRuns,
            "succ": self.totalSucc,
        }
        h5.write_dict("manifest", manifest)
        h5.write_dict("params", ry.get_params())
        h5.write_dict("config", self.config_dict)
        h5.write("q", q)
        h5.write_list("contacts", contacts, dtype="int32")
        h5.write_list("force_pairs", pairs, dtype="int32")
        h5.write_list("forces", forces)


# @hydra.main(version_base=None, config_path="../configs", config_name="tra2ctrl")
def main():
    cfg = omegaconf.OmegaConf.load("config.yaml")

    # process cfg: use scenario-specific settings
    process_cfg(cfg)

    # communicate config to underlying c++ lib
    ry.set_params(dict(cfg.ConstrainedSampling.ry))

    RC = StableConfigSampler(cfg.ConstrainedSampling)
    RC.run()


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    main()
