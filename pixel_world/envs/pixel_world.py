import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))
from env_utils import *
import numpy as np

import matplotlib.pyplot as plt

class PixelWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,reward_mapping,world_map="maps/room1.txt",default_state=' ',agent_color=[50,205,50]):
        self.raw_map = []
        self.agent_color = agent_color
        self.initial_state = None
        self.default_state = default_state
        with open(world_map,"r") as f:
            for i,line in enumerate(f.readlines()):
                acc = []
                for j,s in enumerate(line.strip()):
                    state = DiscreteState(**reward_mapping[s],coords=np.array([i,j]))
                    acc.append(state)
                    if s == "S":
                        self.initial_state = state
                self.raw_map.append(acc)
        self.dim = (len(self.raw_map),len(self.raw_map[0]))
        self.current_state = self.initial_state
        self.action_vectors = np.array([[-1,0],[0,1],[1,0],[0,-1]])
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0,high=255,shape=(3,self.dim[0],self.dim[1]))
        
    def _map2screen(self,transpose=False):
        pixel_map = []
        for i in range(self.dim[0]):
            acc = []
            for j in range(self.dim[1]):
                acc.append(self.raw_map[i][j].color if self.raw_map[i][j] != self.current_state else self.agent_color)
            pixel_map.append(acc)
        return np.array(pixel_map) if not transpose else np.array(pixel_map).transpose((2,0,1))
    
    def _action2vec(self,action):
        return self.action_vectors[action]
    
    def _project(self,state):
        i = max(0,min(self.dim[0],state[0])) # Find new (i,j) coordinates but without the agent falling
        j = max(0,min(self.dim[1],state[1]))
        next_state = self.raw_map[i][j] # If the state is not accessible (e.g. wall), return -1 and stay in place
        return next_state if next_state.accessible else -1

    def step(self, action):
        action = self._action2vec(action)
        next_state = self._project(self.current_state.coords + action)
        self.current_state = self.current_state if next_state == -1 else next_state
        reward = self.current_state.get_reward()
        return self._map2screen(True),reward,int(self.current_state.terminal),{} 
    
    def reset(self):
        self.current_state = self.initial_state
        return self._map2screen(True)
    
if __name__ == "__main__":
    env = PixelWorld(navigation_alphabet(),"../../maps/room1.txt")
    print(env.current_state.coords)
    env.step(0)
    env.reset()
    print(env.current_state.coords)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_screen(env,ax)