# Combining Constrained Sampling and Reinforcement Learning for Manipulation (CoRL 2026 paper)

This repo is accompanying the CSRL paper accepted at CoRL 2026.
See the [project page](https://www.user.tu-berlin.de/mtoussai/26-CSRL/) for videos,
and [arXiv](https://arxiv.org/abs/2602.08557)
and [openreview.net](https://openreview.net/forum?id=cJGMjQlgz5) to access the paper.


## Installation

* Clone this repo
```shell
git clone https://github.com/LIS-TU-Berlin/csrl.git
```
* Download constrained samples data from hf (https://huggingface.co/datasets/MarcToussaint/csrl-data) (`pip install huggingface-hub`):
```shell
hf download MarcToussaint/csrl-data --repo-type dataset --local-dir ./data
```
* Install locally, including dependencies
```shell
uv pip install -e .
```

## Usage

### Exploring the scenarios

```shell
ry-view scenes/allegro_cube.yml
```

The `scenes/` folder includes descriptions of the six scenarios.
The were designed using the [robotic package](https://github.com/MarcToussaint/robotic), which also provides the `ry-view` cmd line tool.
Whenever an env is created, they are translated to mujoco mjcf -- for reference, the `scenes/` includes these translations, but with hardcoded paths that will not work on your side.
After either fixing these paths, or once using the EnvCreator (see below), the following should also work:

```shell
python -m mujoco.viewer --mjcf=scenes/allegro_cube_conv.xml
```

### Exploring the constrained samples

```shell
python scripts/explore_data.py
```

The `explore_data.py` includes two examples to display a dataset of constrained samples.
The first is a minimalistic example on how to load the data (into a dict of arrahs).
The second creates an envs and then an CSRL_Resetter, but then uses these only to display the configs loaded by the resetter (plus additional info on contacts, forces, and nearest neighbors).

**NOTE:** If you want to explore using CPU only (disable `warp`) edit `Gym.warp_worlds: 0` in `config.yaml`.


### Random rollouts in the envs

```shell
python scripts/rollout_random_actions.py
```

The script includes two examples:
1) Only creating the mujoco sim (no full gym environment, just the mujocy sim instance), and directly setting control buffers and unrolling.
2) Creating a full gym environment and unrolling random actions.
Doing the latter for action_scale=0 is interesting, as it shows whether the initial state is really static. E.g., for the allego scenarios, translating contact forces to a control reference offset is essential to have zero actions via the PD motors generate the static forces (see appendix in paper).

### Running CSRL (reproducing paper experiments)

The `scripts/CSRL.py` is the main script used for training in the paper.
The `scripts/_runs.sh` shows examples of launching batches of experiments for the different tables in the paper.

### Using the envs directly with sb3 (not efficient, not tuned)

This is a minimal example of using the envs with stablebaselines3.
However, this is not at all tuned.
The current choice of PPO requires different hyperparameters (more warp_worlds, etc).
The CSRL script also shows using TD3 and PPO (within CSRL), but is more involved.

### Generating new constrained samples

For the CSRL publication, we used the cpp implementation in `scripts/cpp_sampler`.
This repo includes a direct python translation `scripts/constrained_sampling.py`.
However, it lacks a few details, including automatic scaling of force DOFs of the NLP (which improves convergence) and adding a center-of-mass constraint for a free floating articulated object (aka humanoid or quadruped).


## Please cite
```bibtex
@InProceedings{toussaint2026combining,
  title		= {Combining Constrained Sampling and Reinforcement Learning for Manipulation},
  author	= {Marc Toussaint and Cornelius V.~Braun and Armand Jordana and Sayantan Auddy and Eckart Cobo-Briesewitz and Denis Shcherba and Tilman Burghoff and Justin Carpentier},
  booktitle	= {Proc{.} of the Annual Conf{.} on Robot Learning (CORL)},
  publisher	= {PMLR},
  year		= {2026},
  url       = {https://openreview.net/forum?id=cJGMjQlgz5},
  webpage	= {https://www.user.tu-berlin.de/mtoussai/26-CSRL/},
  arxiv		= {2602.08557}
}
```
