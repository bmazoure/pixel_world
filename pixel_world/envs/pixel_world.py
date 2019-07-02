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
    """
    Reads a gridworld environment either from a file or from a string.
    """
    def __init__(self,source,source_type,symbols):
        """
        Args:
            source: Either filename or map of the gridworld
            source_type: 'file' or 'string'
        """
        self.memory = []
        self.source = source
        self.source_type = source_type
        self.symbols = set(list(symbols) + ['\n'])

    def read(self):
        if self.source_type == 'file':
            f = open(self.source,"r")
            self.memory = [line.strip() for line in f.readlines()]
            f.close()
        else:
            tmp = ''.join(filter(self.symbols.__contains__, self.source))
            self.memory = tmp.split('\n')
            for i in range(len(self.memory)):
                self.memory[i] = self.memory[i].strip()
            self.memory = list(filter(None,self.memory))
        return self.memory

    def close(self):
        self.memory = []
        
class PixelWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,reward_mapping,world_map="maps/room1.txt",default_state=' ',from_string=False,state_type="xy",actions='2d_discrete',channels_first=True,max_steps=30):
        """
        Main class of a pixel world type. Everything is a Discrete State object, with (x,y) coordinates saved, but RGB observations can be returned instead.
        Args:
            reward_mapping: Dictionary which maps alphabet symbols (e.g. #, 0, S, etc) defined by the user onto a set of properties for every state type.
            world_map: This can be either a file name (with from_string=False), or a text map following the correct semantics (with from_string=True).
            default_state: Default state to be used in case the map contains unknown symbols.
            from_string: Whether the world_map is a filename or a map.
            state_type: Either the true state (x,y coordinats in this case), or observations (RGB output).
            actions: Of the following types: 1d_discrete (up, right, down,left), 1d_horizontal (left, right), 1d_vertical (up, down), 2d_continuous (2d vector between [-1,1]).
            channels_first (bool): If True, tranposes channels to (3,h,w) instead of (h,w,3).
            max_steps (int): Number of maximum episode length.
        """
        self.state_type = state_type
        self.actions = actions
        self.channels_first = channels_first
        self.max_steps = max_steps
        self.raw_map = []
        self.agent_color = reward_mapping['.agent']['color']
        self.initial_states = []
        self.goal_states = []
        self.default_state = default_state
        self.accessible_states = []
        self.reward_mapping = reward_mapping
        reader = EnvReader(source=world_map,source_type='str' if from_string else 'file', symbols=reward_mapping.keys())
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
                    self.initial_states.append(state)
                if reward_mapping[s]['accessible']:
                    self.accessible_states.append(state)
            self.raw_map.append(acc)
        self.dim = (len(self.raw_map),len(self.raw_map[0]))
        self.current_state = np.random.choice(self.initial_states,1)[0]
        if self.actions == '2d_discrete':
            self.action_vectors = np.array([[-1,0],[0,1],[1,0],[0,-1]])
            self.action_space = spaces.Discrete(4)
        elif self.actions == '1d_horizontal':
            self.action_vectors = np.array([[0,1],[0,-1]])
            self.action_space = spaces.Discrete(2)
        elif self.actions == '1d_vertical':
            self.action_vectors = np.array([[1,0],[-1,0]])
            self.action_space = spaces.Discrete(2)
        if self.actions == '2d_continuous':
            low = np.array([-1,-1])
            high = np.array([1,1])
            self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
        if self.state_type == 'image':
            self.observation_space = spaces.Box(low=0,high=255,shape=(3,self.dim[0],self.dim[1]),dtype=np.uint8)
        elif self.state_type == "xy":
            low = np.array([0,0])
            high = np.array([len(self.raw_map)-1,len(self.raw_map[0])-1])
            self.observation_space = spaces.Box(low=low,high=high,dtype=np.uint8)
        self.visited = []
        self.current_steps = 0
        
    def _map2screen(self):
        pixel_map = []
        for i in range(self.dim[0]):
            acc = []
            for j in range(self.dim[1]):
                acc.append(self.raw_map[i][j].color if self.raw_map[i][j] != self.current_state else self.agent_color)
            pixel_map.append(acc)
        return np.array(pixel_map) if not self.channels_first else np.array(pixel_map).transpose((2,0,1))
    
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
        s_p_a = self.current_state.coords
        next_s_p_a = self.current_state.coords + action
        next_s_p_a, next_state = self._project(next_s_p_a)
        
        if next_state != -1:
            self.current_state = next_state
            s_p_a = next_s_p_a

        if self.current_state.collectible and self.current_state in self.visited: # if collectible, reset reward back to default
            reward = self.reward_mapping[self.default_state]['reward_pdf']()
        else: 
            reward = self.current_state.get_reward()

        self.visited.append(self.current_state)
        self.current_steps += 1
        is_terminal = int(self.current_state.terminal)
        is_done = (self.current_steps > self.max_steps) or is_terminal
        next_obs = s_p_a if self.state_type != 'image' else self._map2screen()
        return next_obs,reward,is_done,{} 

    def reset_to_state(self,state):
        for s in self.accessible_states:
            if np.all(s.coords == state):
                self.current_state = s
        self.visited = []
        self.current_steps = 0
    
    def reset(self):
        self.current_state = np.random.choice(self.initial_states,1)[0]
        next_obs = self.current_state.coords if self.state_type != 'image' else self._map2screen()
        self.visited = []
        self.current_steps = 0
        return next_obs
    
if __name__ == "__main__":
    env = PixelWorld(navigation_alphabet(),"####\n#S #\n## #\n####",from_string=True,actions="2d_discrete")
    print(env.current_state.coords)
    env.step(1)
    print(env.current_state.coords)
    env.reset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
