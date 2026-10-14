import omegaconf
import torch

from csrl.envs.env_creator import EnvCreator
from csrl.RL.CSRL_resetter import CSRL_Resetter


def play_random_controls(cfg):
    creator = EnvCreator(cfg)
    sim = creator.create_Sim(creator.C, cfg.Gym.num_scenes, cfg.Gym.warp_worlds)
    sim.view_speed = 1.0
    for t in range(100):
        ctrl_buffer = 0.1 * torch.randn((20, sim.model.nu))
        sim.rollout(ctrl_buffer)
        print(t)


def play_random_actions(cfg, action_scale=1.0):
    creator = EnvCreator(cfg)
    env = creator.create_GridGym(1, 1)
    resetter = CSRL_Resetter(creator.C, creator.goal_feat_map, creator.get_config_embedding, creator.cfg.CSRL)
    starts, starts_r, goals, goals_feat = resetter.getStartsGoals_forTesting(10, verbose=0)
    env.set_starts_goals(starts, starts_r, goals, goals_feat)
    pi = lambda x, t: action_scale * torch.randn(env.num_scenes, env.action_dim)
    creator.sim.view_speed = 1.0
    env.verbose = 0
    env.cfg.time_limit = 2.0
    for _ in range(10):
        env.rollout(pi, False)


if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load("config.yaml")

    # play_random_controls(cfg)
    play_random_actions(cfg, action_scale=0.0)
