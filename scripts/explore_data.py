import time

import numpy as np
import omegaconf
import robotic as ry
from robotic.tools import h5_helper, video_gen

from csrl.envs.env_creator import EnvCreator
from csrl.RL.CSRL_resetter import CSRL_Resetter


def load_and_display():
    """minimal low-level example to load and display configuration data"""

    datafile = "data/configs/double_sphere_Sep30.h5"
    scene_yml = "scenes/double_sphere.yml"

    data = h5_helper.H5Reader(datafile).read_all()
    print("-- dataset entries:", *data.keys())
    q = data["q"]

    C = ry.Config()
    C.addFile(scene_yml)
    for i in range(q.shape[0]):
        C.setJointState(q[i])
        C.view(True, f"config {i}")


def browse_configs(n=30, pause=False, show_nearest=False, make_video=False):
    """essentially the same, but using the CSRL_Resetter and showing more info"""
    creator = EnvCreator(cfg=omegaconf.OmegaConf.load("config.yaml"))
    resetter = CSRL_Resetter(creator.C, creator.goal_feat_map, creator.get_config_embedding, creator.cfg.CSRL)
    resetter.load_configs()
    print("support objects:", resetter.supportObjs, resetter.configs.supports)

    # ry.set_params({"Render/renderText": False, "Render/floorColor": [1.0]}) #"Render/lights": [-3., 2., -3., 3., 0., -4.],

    if make_video:
        V = video_gen.VideoGenerator(input_framerate=50, filename=f"{creator.cfg.scenario}_samples.mp4")

    if show_nearest:
        tree = resetter.create_config_KDTree(resetter.configs)

    id2name = creator.C.getFrameNames()

    if n <= 0:
        n = resetter.configs.q.shape[0]

    for i in range(n):
        q = resetter.configs.q[i]
        f = resetter.configs.forces[i]
        c = resetter.configs.contacts[i]
        creator.C.setJointState(q)

        info = f"config {i}: contacts: {[id2name[i] for i in c]}"
        # info = f'contacts: {['table']+[id2name[i] for i in c[3::2]]}' #
        info += f", forces:\n{f.reshape(-1, 9)}"
        print("==", info)
        z = creator.get_config_embedding(q, None, resetter.supportObjs)
        print(f"-- interp embedding: {z}")

        key = creator.C.view(pause, info)
        # filename = f'{creator.cfg.scenario}_sample_'
        # creator.C.get_viewer().savePng(filename)

        if make_video:
            rgb = creator.C.get_viewer().getRgb()
            # Image.fromarray(rgb).save(f"data/configs/images/{creator.cfg.scenario}_sample_{i:06d}.jpg")
            V.add([rgb] * 25)

        if key == ord("q"):
            break
        time.sleep(0.01)

        if show_nearest:
            d, j = tree.query(z, k=2)
            q = resetter.configs.q[j[1]]
            f = resetter.configs.forces[j[1]]
            c = resetter.configs.contacts[j[1]]
            creator.C.setJointState(q)
            info = f"== NEAREST config {j}, distance: {d[1]}, contacts: {c} {[id2name[i] for i in c]}"  # , forces:\n{f.reshape(-1,9)}'
            print(info)
            z = creator.get_config_embedding(q, None, resetter.supportObjs)
            print(f"-- NLP feature: {z}")

            key = creator.C.view(pause, info)
            if key == ord("q"):
                break
            time.sleep(0.2)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    # load_and_display()
    browse_configs(n=100, pause=True, show_nearest=False, make_video=False)
