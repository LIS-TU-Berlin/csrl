import math
import os
import time
import traceback

import hydra
import numpy as np
import omegaconf
import torch
from stable_baselines3 import PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from csrl.envs import mujoco_gym, vec_gym
from csrl.envs.env_creator import EnvCreator
from csrl.RL.CSRL_resetter import CSRL_Resetter
from csrl.RL.RL_train_loop import RL_train_loop
from csrl.RL.TD3.agent import TD3_Agent
from csrl.RL.TD7.agent import TD7_Agent
from csrl.RL.writer import Writer


class CustomCallback(BaseCallback):
    def __init__(self, env: mujoco_gym.MujocoGym, rl_method, tag, t_inc_per_it=1, report_its=100, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.it = 0
        self.report_its = report_its
        self.trains_per_it = t_inc_per_it
        self.next_report = report_its
        self.writer = Writer(tag)
        self.rl_method = rl_method
        self.best_reward = -1e10

    def _on_step(self) -> bool:
        self.it += self.trains_per_it

        if self.it >= self.next_report:
            eps = self.env.total_eps.item()

            if eps > 0:
                avg_reward = self.env.total_reward.item() / eps
                if avg_reward > self.best_reward:
                    self.rl_method.save(f"data/tmp/{self.writer.tag}_sb_model.zip")
                    self.best_reward = avg_reward

            self.writer.add("ep/reward", self.env.total_reward, eps)
            self.writer.add("ep/succ", self.env.total_succ, eps)
            self.writer.add("ep/steps", self.env.total_steps, eps)
            self.writer.add("ep/terminated", self.env.total_terminated, eps)
            self.env.total_eps, self.env.total_steps, self.env.total_reward, self.env.total_succ, self.env.total_terminated = 0.0, 0.0, 0.0, 0.0, 0.0

            self.next_report += self.report_its

            self.writer.write(self.it, verbose=1)

        return True


def main(cfg=None):
    if cfg == None:
        cfg = omegaconf.OmegaConf.load("config.yaml")

    # -- delay start? (for hydra multi-runs)
    if hasattr(cfg, "hydra_runs"):
        num_gpus = int(os.environ["CUDA_NUM_GPUS"])
        job_id = hydra.core.hydra_config.HydraConfig.get().job.num
        gpu_id = (job_id + cfg.first_gpu) % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        time.sleep(job_id * 10)
        print(f"===== hydra run {job_id} on gpu {gpu_id}/{num_gpus} =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- create env
    # if cfg.CSRL.T_curr==0 and hasattr(cfg.CSRL[cfg.scenario], 'T_curr'):
    # cfg.CSRL[cfg.scenario].T_curr = 0
    tag_scenario = f"{cfg.scenario}{'_sparse' if cfg.Gym.reward_dense_radius <= 0.0 else ''}"
    tag_mode = f"{cfg.CSRL.start_goal_mode}"  # {"c" if cfg.CSRL.T_curr>0 else ""}'
    tag_RLmethod = f"{cfg.CSRL.RL_method}{'_her' if cfg.CSRL.RL_method == 'TD7' and cfg.TD7.HER_ratio > 0.0 else ''}"
    creator = EnvCreator(cfg, f"-{tag_scenario}-{tag_RLmethod}-{tag_mode}")
    print("-- tag:", cfg.tag)
    if cfg.Gym.num_scenes > 1:
        env = creator.create_GridGym(cfg.Gym.num_scenes, 1)
    else:
        # creator.create_GridGym(1,1)
        env = creator.create_Gym()
    # creator.sim.view_speed = 1.
    gamma = math.pow(0.5, env.cfg.tau_step / creator.cfg.CSRL.discount_half_life)
    print("-- discount gamma:", gamma)

    # -- create RL method
    sb = False
    if cfg.CSRL.RL_method in ["TD3", "TD7"]:
        cfg.TD7.tag = cfg.tag
        cfg.TD7.discount = gamma
        state_dim = env.observation_space.shape[-1]
        action_dim = env.action_space.shape[-1]
        max_action = float(np.max(env.action_space.high))
        if cfg.CSRL.RL_method == "TD7":
            rl_agent = TD7_Agent(state_dim, action_dim, max_action, device, offline=False, cfg=cfg.TD7)
        elif cfg.CSRL.RL_method == "TD3":
            rl_agent = TD3_Agent(env, device, cfg.TD7)
        rl_method = RL_train_loop(env, rl_agent, device, cfg.TD7)
        if cfg.CSRL.initialize_agent != "none":
            print(f"-- initializing with model {cfg.CSRL.initialize_agent}")
            rl_method.agent.hp.datasize_before_execution = -1
            rl_method.agent.hp.datasize_before_training = rl_method.agent.hp.buffer_size / 4
            rl_method.agent.load(cfg.CSRL.initialize_agent)
        writer = rl_method.writer.writer
    elif cfg.CSRL.RL_method in ["sbTD3", "sbTD3_her"]:
        sb_env = vec_gym.SB_VecGym(env)
        n_actions = env.single_action_space.shape[0]
        policy_kwargs = dict(net_arch=[256, 256])  # , activation_fn=torch.nn.LeakyReLU)
        rl_method = TD3(
            "MlpPolicy",
            sb_env,
            gamma=gamma,
            # policy_delay=cfg.TD7.policy_freq,
            learning_rate=cfg.TD7.learning_rate,
            buffer_size=cfg.TD7.buffer_size,
            learning_starts=cfg.TD7.datasize_before_training,
            batch_size=cfg.TD7.batch_size,
            train_freq=1,
            gradient_steps=cfg.TD7.train_steps,
            action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=cfg.TD7.exploration_noise * np.ones(n_actions)),
            target_policy_noise=cfg.TD7.target_policy_noise,
            target_noise_clip=cfg.TD7.noise_clip,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )  # , tensorboard_log="./tensorboard/")

        if cfg.CSRL.RL_method == "sbTD3_her":
            her_env = vec_gym.SB_VecGoalGym(env)
            rl_method = TD3(
                "MultiInputPolicy",
                her_env,
                replay_buffer_class=HerReplayBuffer,
                # policy_delay=cfg.TD7.policy_freq,
                learning_rate=cfg.TD7.learning_rate,
                buffer_size=cfg.TD7.buffer_size,
                learning_starts=cfg.TD7.datasize_before_training,
                batch_size=cfg.TD7.batch_size,
                action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=cfg.TD7.exploration_noise * np.ones(n_actions)),
                target_policy_noise=cfg.TD7.target_policy_noise,
                target_noise_clip=cfg.TD7.noise_clip,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )  # , tensorboard_log="./tensorboard/")
            rl_method.replay_buffer.copy_info_dict = True

        sb = True
        callback = CustomCallback(env, rl_method, cfg.tag, report_its=cfg.TD7.eval_freq, t_inc_per_it=cfg.TD7.train_steps)
        writer = callback.writer.writer
    elif cfg.CSRL.RL_method == "sbSAC":
        assert env.num_scenes == 1
        rl_method = SAC("MlpPolicy", env, gamma=gamma, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=3e-4, verbose=1, tensorboard_log="./tensorboard/")
        callback = CustomCallback(env, rl_method, cfg.tag, report_its=cfg.TD7.eval_freq)
    elif cfg.CSRL.RL_method == "sbPPO":
        sb_env = vec_gym.SB_VecGym(env)
        n_actions = env.single_action_space.shape[0]
        policy_kwargs = dict(net_arch=[256, 256])  # , activation_fn=torch.nn.LeakyReLU)
        rl_method = PPO(
            "MlpPolicy",
            sb_env,
            gamma=gamma,
            n_steps=cfg.PPO.n_steps,  #!!! assumes we have highly parallelized the envs
            batch_size=cfg.PPO.batch_size,  # cfg.TD7.batch_size, !!! much less than for TD7!
            n_epochs=cfg.PPO.n_epochs,  # training steps per iteration
            learning_rate=cfg.PPO.learning_rate,  # cfg.TD7.learning_rate,
            # clip_range_vf = cfg.TD7.value_clip[1],
            policy_kwargs=policy_kwargs,
            verbose=0,
        )  # , tensorboard_log="./tensorboard/")
        # t_inc_per_it = cfg.PPO.n_epochs * (env.num_scenes//cfg.PPO.batch_size)
        t_inc_per_it = cfg.TD7.train_steps
        callback = CustomCallback(env, rl_method, cfg.tag, report_its=cfg.PPO.eval_freq, t_inc_per_it=t_inc_per_it)
        writer = callback.writer.writer
        sb = True
    else:
        raise Exception(f"RLalgo not defined: {cfg.CSRL.RL_method}")

    # -- setup CSRL resetter
    cfg.CSRL.files = cfg.files
    resetter = CSRL_Resetter(creator.C, creator.goal_feat_map, creator.get_config_embedding, cfg.CSRL)
    if "rv" in resetter.cfg.start_goal_mode:
        resetter.set_reverse_curriculum_helper_gym(creator.create_GridGym(1, 1, warp_worlds=0))

    # -- training loop, updating start-goals in each epoch
    start_time = time.time()
    t = 0
    while t < cfg.CSRL.T_end:
        resetter.update_environment(env, t, rl_method)
        if sb:
            # we want to perform T_block training steps
            # if cfg.CSRL.RL_method=='sbPPO':
            #     trains_per_it = cfg.PPO.n_epochs * (env.num_scenes//cfg.PPO.batch_size)
            #     env_steps = cfg.CSRL.T_block * env.num_scenes / trains_per_it
            # else:
            # this is the number of env steps that TD7-reference does in each block:
            env_steps = cfg.CSRL.T_block * 2000 / 20  # env.num_scenes / cfg.TD7.train_steps
            rl_method.learn(total_timesteps=env_steps, reset_num_timesteps=False, callback=callback)
        else:
            rl_method.train(train_steps=cfg.CSRL.T_block)
        t += cfg.CSRL.T_block

    # -- final evaluation
    resetter = CSRL_Resetter(creator.C, creator.goal_feat_map, creator.get_config_embedding, cfg.CSRL)
    starts, starts_r, goals, goal_feats = resetter.getStartsGoals(n=2 * cfg.CSRL.final_eval_episodes, alpha=1.0, mode="CC")
    env.set_starts_goals(starts, starts_r, goals, goal_feats)
    env.cfg.time_limit = 3.0 * cfg.CSRL.nominal_time_limit
    env.reset()
    succ, reward, terminated, steps = resetter.evaluate_policy(
        env, rl_method, num_episodes=cfg.CSRL.final_eval_episodes, view=False, make_video=False, collect_histograms=False, verbose=1
    )

    # -- write report
    writer.add_scalar("ep/succ", succ, t + cfg.CSRL.final_eval_episodes)

    report = f"{cfg.tag:50} , {tag_scenario} , {tag_RLmethod:9} , {tag_mode:4} , {succ:5.3f} , {reward:5.3f} , {terminated:5.3f} , {steps:5.2f} , {time.time() - start_time:7.2f}, {cfg.seed}\n"

    print("====", report)
    with open(f"data/tmp/{cfg.tag}_run.csv", "w") as fil:
        fil.write(report)
    with open("data/tmp/runs.csv", "a") as fil:
        fil.write(report)

    return float(succ)


@hydra.main(version_base=None, config_path="..", config_name="config")
def hydra_main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    try:
        main(cfg)
    except Exception as e:
        print("~~~~~~~~~~ exception ~~~~~~~~~~")
        print(traceback.format_exc())
        print(e)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    hydra_main()
