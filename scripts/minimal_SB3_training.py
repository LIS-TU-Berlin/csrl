import omegaconf
from stable_baselines3 import PPO

from csrl.envs.env_creator import EnvCreator
from csrl.envs.vec_gym import SB_VecGym
from csrl.RL.CSRL_resetter import CSRL_Resetter


# @hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg=None):
    if cfg == None:
        cfg = omegaconf.OmegaConf.load("config.yaml")

    # -- create env
    creator = EnvCreator(cfg)
    env = creator.create_Gym()

    # -- create RL method
    vec_env = SB_VecGym(env)  # env is natively vectorized, this just wraps it with SB3 conventions
    rl_method = PPO("MlpPolicy", vec_env, gamma=creator.get_gamma(), policy_kwargs=dict(net_arch=[256, 256]), learning_rate=3e-4, verbose=1, tensorboard_log="./tensorboard/")

    # -- training in epochs, where new start-goals are selected for each epoch
    resetter = CSRL_Resetter(creator.C, creator.goal_feat_map, creator.get_config_embedding, creator.cfg.CSRL)
    t = 0
    while t < cfg.CSRL.T_end:
        # before using the env, one has to set starts (incl start control refs) and goals (incl goal_features):
        starts, starts_ref, goals, goals_feat = resetter.getStartsGoals(n=10000, alpha=1.0, mode="CC")
        env.set_starts_goals(starts, starts_ref, goals, goals_feat)
        rl_method.learn(total_timesteps=env.num_scenes * cfg.CSRL.T_block, reset_num_timesteps=False, log_interval=100)
        t += cfg.CSRL.T_block


if __name__ == "__main__":
    main()
