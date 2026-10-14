import os
import random

import numpy as np
import omegaconf
import torch
from robotic.tools import h5_helper


class H5MatchFile:
    def __init__(self, h5_filename):
        h5 = h5_helper.H5Reader(h5_filename)
        self.__dict__ = h5.read_all()  # MAGIC!


def set_global_seeds(seed: int, deterministic=False):
    """Seed Python, NumPy, PyTorch, and robotic-py for reproducible runs.

    `seed` is required. Set `deterministic=True` to additionally enable
    PyTorch's deterministic mode.
    """

    torch.set_float32_matmul_precision("high")
    assert seed is not None
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # and pick_device().type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # required for deterministic CUDA matmul under use_deterministic_algorithms
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        import robotic as ry

        ry.rnd_seed(seed)
    except Exception:
        pass


def set_env_seed(env, seed: int):
    """Seed a gymnasium env's action/observation spaces.

    `Space.sample()` (e.g. `env.action_space.sample()`) uses the space's own
    RNG, independent of global numpy.
    """
    assert seed is not None
    for attr in (
        "action_space",
        "observation_space",
        "single_action_space",
        "single_observation_space",
    ):
        space = getattr(env, attr, None)
        if space is not None and hasattr(space, "seed"):
            try:
                space.seed(seed)
            except Exception:
                pass


def process_cfg(cfg):
    # -- overwrite scenario-specific settings, then save:
    # parse sub-sub-dicts: if specific scenario -> copy to sub-dict; then delete them all (except ry)
    for k, v in cfg.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            for kk, vv in v.items():
                if isinstance(vv, omegaconf.dictconfig.DictConfig):
                    if kk == cfg.scenario:  # copy them to sub-dict v
                        for kkk, vvv in vv.items():
                            v[kkk] = vvv
                    if kk != "ry":
                        v.__delattr__(kk)  # delete sub-sub-dict

    os.system("mkdir -p data/tmp")
    omegaconf.OmegaConf.save(config=cfg, f=f"data/tmp/{cfg.tag}.yaml")
    return cfg
