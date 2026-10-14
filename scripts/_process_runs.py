# from here: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
# or here: https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
# or here: https://www.tensorflow.org/tensorboard/dataframe_api

import math

import numpy as np
import omegaconf
import pandas
from robotic.tools.h5_helper import H5Reader, H5Writer
from robotic.tools.plot_helper import PlotHelper
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from csrl.creator import EnvCreator
from csrl.CSRL_resetter import CSRL_Resetter
from csrl.TD7.agent import TD7_Agent


def tensorboard2dict(path_names, tags):
    data = {}
    # manifest = {}
    shared_steps = None

    #        if tags=='ALL':
    #            tags = event_acc.Tags()["scalars"]

    for tag in tags:
        tag_name = tag.replace("/", "_")
        data[tag_name] = []

    length = None

    for path in path_names:
        # print('-- reading tensorboard:', path)
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        for tag in tags:
            # print('== tag:', tag)
            event_list = event_acc.Scalars(tag)
            values = np.array([x.value for x in event_list])
            steps = np.array([x.step for x in event_list])
            # print('== values:', values)
            # print('== steps:', steps)

            tag_name = tag.replace("/", "_")
            data[tag_name].append(values)
            if shared_steps is None:
                shared_steps = steps
                length = steps.size
            else:
                length = min(length, steps.size)
                assert np.all(shared_steps[:length] == steps[:length])

    for tag in tags:
        tag_name = tag.replace("/", "_")
        if len(data[tag_name]) == 0:
            data[tag_name] = np.zeros(0)
        else:
            data[tag_name] = np.stack([d[:length] for d in data[tag_name]])

    if length is None:
        data["steps"] = np.zeros(0)
    else:
        data["steps"] = shared_steps[:length]

    return data


# def grab_cfg_from_run(run):
#     cfg_name = run.replace('TD7', 'cfg')
#     with open(f'data/tmp/{cfg_name}.yaml', 'r', encoding='utf-8') as fil:
#         data = yaml.safe_load(fil)

#     scenario = data['scenario']
#     time_limit = data['Gym']['time_limit']
#     mode = data['CSRL']['start_goal_mode']
#     if 'start_goal_update_steps' in data['CSRL']:
#         T_block = data['CSRL']['start_goal_update_steps']
#         T_schedule = data['CSRL']['start_goal_direct_step']
#         T_end = data['CSRL']['schedule_until_step']
#     else:
#         T_block = data['CSRL']['T_block']
#         T_schedule = data['CSRL']['T_schedule']
#         T_end = data['CSRL']['T_end']
#     T_cloning = data['CSRL']['T_cloning']

#     # print(scenario, time_limit, mode, update_steps, sched_until, direct_step, train_steps, gamma)

#     # name = f'{scenario}_{mode}_s{(1 if sched_until>0 and update_steps>0 else 0)}_bc{1 if clone_until>0 else 0}'
#     name = mode.replace('direct', 'baseline')
#     if T_end>0 and T_block>0:
#         name += 'Sched'
#     if T_cloning>0:
#         name += 'BC'
#     print(name)
#     return name

# ORDER = ['traj', 'trajSched', 'trajSchedBC', 'interp', 'interpSched', 'baseline']

# def conv_runs(runs):
#     names = [f'{i//6}_{grab_cfg_from_run(run)}' for i,run in enumerate(runs)]
#     paths = [f'tensorboard/{run}' for run in runs]

#     path_names = []
#     for o in ORDER:
#         for p,n in zip(paths,names):
#             if n[2:]==o:
#                 path_names.append((p,n))

#     data, manifest = tensorboard2dict(path_names, ['RL/reward_sum_ep'])
#     h5 = data_tools.H5Writer('data/tmp/tensorboard_proc.h5')
#     h5.write_dict('manifest', manifest)
#     for key, value in data.items():
#         h5.write(key, value)

#     print('-- converted runs:', len(path_names))


# def plot(filename):
#     h5 = data_tools.H5Reader(filename)
#     info = h5.read_dict('manifest')

#     data = {}
#     for key, value in info.items():
#         print(key)
#         run = int(key[0])
#         label = key[2:]
#         # label = label.replace('_t1_T20_g95', '')
#         # if 'oneFin' in label:
#         #     label = label.replace('oneFingerBall_', '')
#         #     domain = 'Double Sphere'
#         # else:
#         #     label = label.replace('pandaBall_', '')
#         #     domain = 'Panda Sphere'
#         # label = label.replace('trajectories', 'traj')
#         # label = label.replace('_bc0', '')
#         # label = label.replace('_bc1', '_BC')
#         # label = label.replace('_s0', '_noSched')
#         # label = label.replace('_s1', '_sched')
#         if label not in data:
#             data[label] = {'x': [], 'y': []}
#         data[label]['x'].append( h5.read(f'{key}/steps') )
#         data[label]['y'].append( h5.read(f'{key}/RL_reward_sum_ep') )

#     P = PlotHelper()
#     P.y_log=False
#     P.x_max=2_000_000
#     P.figsize=(6,4)
#     P.legend='lower right'
#     P.grid=True
#     P.quant=[(.25, .2)]
#     if 'double_sphere' in filename:
#         title = 'Double Sphere'
#         P.legend='lower right'
#     else:
#         title = 'Panda Sphere'
#         P.legend='upper center'
#     P.begin(title, 'steps', 'avg. sum of reward')

#     for key, value in data.items():
#         X = value['x']
#         Y = value['y']
#         min_len = min([x.shape[0] for x in X])
#         X = [x[:min_len] for x in X]
#         Y = [y[:min_len] for y in Y]
#         X = np.stack(X)
#         Y = np.stack(Y)

#         mu = P._mean(Y)
#         sig = P._std(Y)
#         _lo_up = [(mu-sig, mu+sig, .15)]
#         # _lo_up = P._lo_ups(Y)
#         P.add(key, P._mean(X), P._mean(Y), _lo_up)

#     P.end(f'{title.replace(' ','_')}.pdf', show=True)


def reevaluate_run(tag, seed):
    print(f"=== reevaluating tag {tag} ===")
    config_file = f"data/tmp/{tag}.yaml"
    cfg = omegaconf.OmegaConf.load(config_file)
    org_seed = cfg.seed

    tag_scenario = f"{cfg.scenario}{'_sparse' if cfg.Gym.reward_dense_radius <= 0.0 else ''}"
    tag_mode = f"{cfg.CSRL.start_goal_mode}"
    tag_RLmethod = f"{cfg.CSRL.RL_method}{'_her' if cfg.TD7.HER_ratio > 0.0 else ''}"
    # if cfg.CSRL.T_curr>0 and tag_mode[-1]!="c":
    #     tag_mode = tag_mode+"c"

    cfg.files.agent = f"data/tmp/{tag}.torch"
    cfg.Gym.warp_worlds = 1000
    cfg.TD7.compile = False
    cfg.seed = seed

    creator = EnvCreator(cfg)
    env = creator.create_Gym()
    resetter = CSRL_Resetter(creator.C, creator.goal_feat_map, creator.get_config_embedding, creator.cfg.CSRL)
    starts, starts_r, goals, goal_feats = resetter.getStartsGoals(n=10_000, alpha=1.0, mode="CC")
    env.set_starts_goals(starts, starts_r, goals, goal_feats)
    env.cfg.time_limit = 5.0 * cfg.CSRL.nominal_time_limit

    state_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    max_action = float(np.max(env.action_space.high))
    rl_agent = TD7_Agent(state_dim, action_dim, max_action, device=env.device, offline=False, cfg=cfg.TD7)
    rl_agent.load(cfg.files.agent)

    succ, reward, terminated, steps = resetter.evaluate_policy(env, rl_agent, num_episodes=1000, view=False, make_video=False, collect_histograms=False, verbose=1)
    report = f"{tag:50} , {tag_scenario} , {tag_RLmethod:9} , {tag_mode:4} , {succ:5.3f} , {reward:5.3f} , {terminated:5.3f} , {steps:5.2f} , {0}, {org_seed}\n"

    print("====", report)
    with open("data/tmp/runs_reeval.csv", "a") as fil:
        fil.write(report)


def reevaluate_runs(runs_file):
    with open(runs_file, "r") as f:
        tags = eval(f.read())

    print(tags)
    n = len(tags)
    for i, tag in enumerate(tags):
        print(f"========== {100.0 * i / n:4.1f}% ({i}/{n}): {tag} ==========")
        reevaluate_run(tag, seed=i)


base_method = "TD7_her"

tab1_scenarios = ["double_sphere", "double_sphere_sparse", "panda_sphere", "panda_sphere_sparse"]
tab1_modes = ["RR", "RRi", "RRic", "RC", "RCi", "RCic", "CC", "CCi", "CCic"]

tab2_scenarios = ["double_sphere", "panda_sphere"]  # , 'sphere_cube', 'panda_cube']
# tab2_methods = ['TD7', 'sbTD3', 'TD7_her', 'sbTD3_her']
tab2_methods = ["TD7_her", "TD7", "sbTD3"]  # , 'sbPPO']
# tab2_modes = ['RRic', 'RCic', 'CCic']
tab2_modes = ["CC", "CCi", "CCic"]

# scenarios1 = ['double_sphere', 'panda_sphere', 'sphere_cube', 'panda_cube']
tab3_scenarios = ["sphere_cube", "panda_cube", "allegro_cube", "allegroDown_cube"]
tab3_modes = ["CC", "CCi", "CCic"]

tab4_scenarios = ["double_sphere", "panda_sphere"]
tab4_modes = ["rvRc", "rvCc"]

plot_scenarios = ["double_sphere", "double_sphere_sparse", "panda_sphere", "panda_sphere_sparse", "sphere_cube", "panda_cube", "allegro_cube", "allegroDown_cube"]
plot_modes = ["RR", "RRi", "RRic", "RC", "RCi", "RCic", "CC", "CCi", "CCic"]


def generate_tables(csv_file):
    D = pandas.read_csv(csv_file, sep=r"\s*,\s*", engine="python")
    print(D.keys())

    print("\n% table1: COMPARISON of modes\n")

    used_runs = []

    print("mode", end=" ")
    for s in tab1_scenarios:
        print(f"& {s}", end=" ")
    print("\\\\")
    for m in tab1_modes:
        print(f"{m:4}", end=" ")
        for s in tab1_scenarios:
            d = D.loc[(D["scenario"] == s) & (D["RL_method"] == base_method) & (D["start_goal_mode"] == m)]
            used_runs += d["tag"].to_list()
            d = d["succ"]
            print(f"& \\val{{{d.mean():.3f}}}{{{d.std():.3f}}}{{{d.size}}}", end=" ")
        print("\\\\")

    print("\n% table2: COMPARISON of rl methods\n")

    print("mode", end=" ")
    for s in tab2_scenarios:
        print(f"{s}", end=" ")
        for r in tab2_methods:
            print(f"& {r}", end=" ")
    print("\\\\")
    for m in tab2_modes:
        print(f"{m:4}", end=" ")
        for s in tab2_scenarios:
            # print(f'\n%% {s}')
            # for m in sg_modes:
            #     d = D.loc[(D['scenario']==s) & (D['RL_method']=='TD7') & (D['start_goal_mode']==m)]
            #     d = d['succ']
            #     print(f'{m:4} & \\val{{{d.mean():.3f}}}{{{d.std():.3f}}}{{{d.size}}} \\\\',f'%% {d.to_numpy()}')
            # print()

            for rl in tab2_methods:
                d = D.loc[(D["scenario"] == s) & (D["RL_method"] == rl) & (D["start_goal_mode"] == m)]
                used_runs += d["tag"].to_list()
                d = d["succ"]
                print(f"& \\val{{{d.mean():.3f}}}{{{d.std():.3f}}}{{{d.size}}}", end=" ")
        print("\\\\")

    print("\n% table3: COMPARISON advanced\n")

    print("mode", end=" ")
    for s in tab3_scenarios:
        print(f"& {s}", end=" ")
    print("\\\\")
    for m in tab3_modes:
        print(f"{m:4}", end=" ")
        for s in tab3_scenarios:
            d = D.loc[(D["scenario"] == s) & (D["RL_method"] == base_method) & (D["start_goal_mode"] == m)]
            used_runs += d["tag"].to_list()
            d = d["succ"]
            print(f"& \\val{{{d.mean():.3f}}}{{{d.std():.3f}}}{{{d.size}}}", end=" ")
        print("\\\\")

    print("\n% table4: reverse curr\n")

    print("mode", end=" ")
    for s in tab4_scenarios:
        print(f"& {s}", end=" ")
    print("\\\\")
    for m in tab4_modes:
        print(f"{m:4}", end=" ")
        for s in tab4_scenarios:
            d = D.loc[(D["scenario"] == s) & (D["RL_method"] == "TD7_her") & (D["start_goal_mode"] == m)]
            used_runs += d["tag"].to_list()
            d = d["succ"]
            print(f"& \\val{{{d.mean():.3f}}}{{{d.std():.3f}}}{{{d.size}}}", end=" ")
        print("\\\\")

    used_runs = list(set(used_runs))
    used_runs.sort()
    print("== used runs: ==\n", len(used_runs))
    with open("z.used_runs", "w") as fil:
        fil.write(str(used_runs))


def conv_tensorboard(csv_file):
    D = pandas.read_csv(csv_file, sep=r"\s*,\s*", engine="python")
    print(D.keys())

    h5 = H5Writer("data/runs/tensorboard.h5")

    for s in plot_scenarios:
        for m in plot_modes:
            d = D.loc[(D["scenario"] == s) & (D["RL_method"] == base_method) & (D["start_goal_mode"] == m)]
            tags = d["tag"].to_list()
            succ = d["succ"]

            print(f"=== scenario: {s}, mode: {m}, #runs: {len(tags)}")

            path_names = [f"tensorboard/{t}" for t in tags]
            data = tensorboard2dict(path_names, ["ep/succ", "ep/reward"])
            print("    loaded data: steps:", data["steps"].shape, "data:", data["ep_succ"].shape, data["ep_reward"].shape)

            h5.write(f"{s}/{m}/succ", succ)
            h5.write(f"{s}/{m}/tags", succ)
            for key, value in data.items():
                h5.write(f"{s}/{m}/{key}", value)


def plot_tensorboard(h5_file):
    h5 = H5Reader(h5_file)
    data = h5.read_all()

    for s in plot_scenarios:
        P = PlotHelper()
        P.y_log = False
        P.y_max = 1.0
        # P.x_max=2_000_000
        P.figsize = (6, 4)
        P.legend = "lower right"
        P.grid = True
        P.quant = [(0.25, 0.2)]
        title = s
        P.legend = "upper left"
        P.begin(title, "steps/1000", "success rate")

        if "cube" in s:
            P.p_idx = 7
        else:
            P.p_idx = 1

        for m in plot_modes:
            succ = data[f"{s}/{m}/succ"]
            X = data[f"{s}/{m}/steps"]
            Y = data[f"{s}/{m}/ep_succ"]
            if X.size > 0:
                print("  adding to plot:", s, m, X[-1], " succ:", np.mean(succ))
                for i in range(succ.size):
                    mu = np.mean(Y[i, -10:-1])
                    # print(s, m, mu, succ[i], X[-1])
                    Y[i, :] *= succ[i] / mu

                mu = np.mean(Y, axis=0)
                sig = np.std(Y, axis=0) / math.sqrt(5)
                _lo_up = [(mu - sig, mu + sig, 0.15)]
                # _lo_up = P._lo_ups(Y)
                P.add(m.replace("RC", "CR"), X / 1000.0, P._mean(Y), _lo_up)

        P.end(f"data/plots/{title.replace(' ', '_')}.pdf", show=True)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    generate_tables("data/runs/runs-26-10-11-reeval.csv")
    # generate_tables('data/runs/runs-26oct.csv')
    # conv_tensorboard('data/runs/runs-26oct.csv')
    # plot_tensorboard('data/runs/tensorboard.h5')

    # reevaluate_runs('data/runs/need_reevaluation.txt')
    # reevaluate_runs('used_runs_noHER')
