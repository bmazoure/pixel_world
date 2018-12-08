# PixelWorld

Gridworld environment made for navigation purposes. Check in the Description section for a detailed overview. 

---
## Installation
To install, do
```
git clone https://github.com/bmazoure/pixel_world.git
cd pixel_world
pip install -e .
```

---
## Description

The repo is structured as follows:
```
maps/
 |- room1.txt
pixel_world/
 envs/
 |-init.py
 |-env_utils.py
```

The environment can built arbitrary defined maps. For example, *room1.txt* is defined as
```
###########
#        0#
#  ##     #
#  ##     #
#S        #
#         #
#         #
###########
###########
```

![](env_screen.png)


The environment has four actions: {**↑**,**→**,**↓**,**←**} corresponding to integers from 0 to 3.

Custom state types are allowed. Below are some default state types:

* **\#**: Wall tile;
* **S**: Initial state;
* **0**: Goal state, gives +R reward and ends the episode. Here, R~|*N*(0,1)|;
* ' ': Empty tile.

Each state is a `DiscreteState` instance with the following attributes:

* **Terminal**: {True/False}, whether the state is terminal, in which case the environment returns `done=True`;
* **Accessible**: {True/False}, whether the agent can access this state. If the state is unaccessible (e.g. a wall), then the agent stays in the previous state;
* **Stochastic**: {True/False}, whether the reward given upon transitionning into this state is sampled from a distribution, or is sampled at the beginning and remains fixed;
* **Reward_pdf**: A callable function which samples rewards from the specified distribution. Below are examples of reward densities:

    * `lambda: np.abs(np.random.normal(loc=1,scale=0.1,size=1)).item()` - |*N*(0,1)|;
    * `lambda: 10` - Dirac at 10;
    * `lambda: np.random.uniform(0,1,size=1)).item()` - |*U*(0,1)|.