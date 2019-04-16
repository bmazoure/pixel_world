import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))
sys.path.append(os.getcwd()+"\\pixel_world\\pixel_world\\")
from env_utils import *
import numpy as np

import matplotlib.pyplot as plt

class EnvReader(object):
    def __init__(self,source,source_type):
        self.memory = []
        self.source = source
        self.source_type = source_type

    def read(self):
        if self.source_type == 'file':
            f = open(self.source,"r")
            self.memory = [line.strip() for line in f.readlines()]
            f.close()
        else:
            self.memory = self.source.split('\n')
        return self.memory

    def close(self):
        self.memory = []
        
class PixelWorldSampler(object):
    def __init__(self,template_env):
        self.template = template_env
        self.map = self.template.text_map
        self.goal_symbol = -1
        for symbol,d in self.template.reward_mapping.items():
            if 'terminal' in d:
                if d['terminal']:
                    self.goal_symbol = symbol
                    for i,line in enumerate(self.map):
                        self.map[i] = line.replace(symbol,self.template.default_state)
        self.accessible_states = self.template.accessible_states
        
    def generate(self,train,train_split,test_split):
        assert train_split + test_split <= 100
        d = lambda x,y:np.linalg.norm(x-y,ord=None)
        agent_coords = self.template.current_state.coords
        unordered_goals = list(filter(lambda x:np.any(x.coords != agent_coords),self.accessible_states))
        # closer <--------> further
        ordered_goals = sorted(unordered_goals,key=lambda x:d(x.coords,agent_coords))
        ordered_goals = unordered_goals ### test for Devon
        th = int(len(ordered_goals) * train_split/100 if train else len(ordered_goals) * test_split/100)
        useful_goals = ordered_goals[:th] if train else ordered_goals[-th:]
        # train_th = len(self.accessible_states) * (len(self.accessible_states)-1)
        # rs = np.random.RandomState(0)
        # idx = rs.permutation(list(range(train_th)))
        # test_th = int(train_th *test_split/100)
        # train_th = int(train_th * train_split/100)
        acc = []
        # mat = self.template.observation_space.sample()
        # for i in range(mat.shape[1]):
        #     for j in range(mat.shape[2]):
        #         mat[:,i,j] = [255,255,255]
        #         if i == 0 or j == 0 or i == mat.shape[1]-1 or j == mat.shape[2]-1:
        #             mat[:,i,j]=[0,0,0]
        #         for s in useful_goals:
        #             if i == s.coords[0] and j == s.coords[1]:
        #                 mat[:,i,j] = [255,125,125]
            
        
        for goal_state in useful_goals:
            x_goal,y_goal = goal_state.coords
            new_map = np.array(self.map,copy=True)
            for i,line in enumerate(new_map):
                for j,c in enumerate(line):
                    if x_goal == i and y_goal == j:
                        new_map[i] = new_map[i][:j] + self.goal_symbol + new_map[i][j+1:]
            acc.append('\n'.join(new_map))
        return np.array(acc)

class PixelWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,reward_mapping,world_map="maps/room1.txt",default_state=' ',from_string=False,as_image=True,actions='2d'):
        self.as_image = as_image
        self.actions = actions
        self.raw_map = []
        self.agent_color = reward_mapping['.agent']['color']
        self.initial_state = None
        self.goal_states = []
        self.default_state = default_state
        self.accessible_states = []
        self.reward_mapping = reward_mapping
        reader = EnvReader(source=world_map,source_type='str' if from_string else 'file')
        self.text_map = reader.read()
        reader.close()
        for i,line in enumerate(self.text_map):
            acc = []
            for j,s in enumerate(line):
                s = s if s in reward_mapping else self.default_state # if we see an unassigned state, make it default
                state = DiscreteState(**reward_mapping[s],coords=np.array([i,j]))
                acc.append(state)
                if reward_mapping[s]['terminal']:
                    self.goal_states.append(state)
                if state.initial: # Note that this overwrites multiple start states to the most recent one
                    self.initial_state = state
                if reward_mapping[s]['accessible']:
                    self.accessible_states.append(state)
            self.raw_map.append(acc)
        self.dim = (len(self.raw_map),len(self.raw_map[0]))
        self.current_state = self.initial_state
        if self.actions == '2d_discrete':
            self.action_vectors = np.array([[-1,0],[0,1],[1,0],[0,-1]])
            self.action_space = spaces.Discrete(4)
        elif self.actions == 'horizontal':
            self.action_vectors = np.array([[0,1],[0,-1]])
            self.action_space = spaces.Discrete(2)
        elif self.actions == 'vertical':
            self.action_vectors = np.array([[1,0],[-1,0]])
            self.action_space = spaces.Discrete(2)
        if self.actions == '2d_continuous':
            low = np.array([-1,-1])
            high = np.array([1,1])
            self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
        if self.as_image:
            self.observation_space = spaces.Box(low=0,high=255,shape=(3,self.dim[0],self.dim[1]),dtype=np.uint8)
        else:
            low = np.array([0,0])
            high = np.array([len(self.raw_map)-1,len(self.raw_map[0])-1])
            self.observation_space = spaces.Box(low=low,high=high,dtype=np.uint8)
        
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
        i = max(0,min(self.dim[0]-1,state[0])) # Find new (i,j) coordinates but without the agent falling
        j = max(0,min(self.dim[1]-1,state[1]))
        next_state = self.raw_map[int(i)][int(j)] # If the state is not accessible (e.g. wall), return -1 and stay in place
        return (i,j),next_state if next_state.accessible else -1

    def step(self, action):
        if self.actions != '2d_continuous':
            action = self._action2vec(action)
        else:
            action = action + np.array([0.5,0.5])
        is_terminal = int(self.current_state.terminal)
        s_p_a = self.current_state.coords + action
        s_p_a, next_state = self._project(s_p_a)
        reward = self.current_state.get_reward()
        self.current_state = self.current_state if next_state == -1 else next_state
        
        next_obs = s_p_a if not self.as_image else self._map2screen(True)
        return next_obs,reward,is_terminal,{} 
    
    def reset(self):
        self.current_state = self.initial_state
        next_obs = self.current_state.coords if not self.as_image else self._map2screen(True)
        return next_obs
    
if __name__ == "__main__":
    env = PixelWorld(navigation_alphabet(),"../../maps/room1.txt")
    print(env.current_state.coords)
    env.step(0)
    env.reset()
    print(env.current_state.coords)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_screen(env,ax)