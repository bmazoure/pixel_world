import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import copy

__all__ = ['navigation_alphabet', 'noisy_navigation_alphabet', 'PixelWorld']

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

    def __init__(self,reward_mapping,world_map="maps/room1.txt",default_state=' ',from_string=False,state_type="xy",actions='2d_discrete',channels_first=True,randomize_goals=False):
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
            randomize_goals (bool): If True, every time reset() is called, pick a goal state at random among all goal states. Else, multi-goal setting
        """
        self.state_type = state_type
        self.actions = actions
        self.channels_first = channels_first
        self.raw_map = []
        self.agent_color = reward_mapping['.agent']['color']
        self.initial_states = []
        self.goal_states = []
        self.default_state = default_state
        self.accessible_states = []
        self.randomize_goals = randomize_goals
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
                    self.goal_states.append(copy.deepcopy(state))
                if state.initial:
                    self.initial_states.append(copy.deepcopy(state))
                if reward_mapping[s]['accessible']:
                    self.accessible_states.append(copy.deepcopy(state))
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
        self.current_map = copy.deepcopy(self.raw_map)
        
    def _map2screen(self):
        pixel_map = []
        for i in range(self.dim[0]):
            acc = []
            for j in range(self.dim[1]):
                acc.append(self.agent_color if (i == self.current_state.coords[0] and  j == self.current_state.coords[1]) else self.current_map[i][j].color)
            pixel_map.append(acc)
        return np.array(pixel_map) if not self.channels_first else np.array(pixel_map).transpose((2,0,1))
    
    def _action2vec(self,action):
        return self.action_vectors[action]
    
    def _project(self,state):
        i = max(0,min(self.dim[0]-1,state[0])) # Find new (i,j) coordinates but without the agent falling
        j = max(0,min(self.dim[1]-1,state[1]))
        next_state = self.current_map[int(i)][int(j)] # If the state is not accessible (e.g. wall), return -1 and stay in place
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
        reward = self.current_state.get_reward()

        if self.current_state.collectible: # if collectible, reset reward back to default
            i,j = next_s_p_a
            self.current_map[i][j] = DiscreteState(**self.reward_mapping[self.default_state],coords=np.array([i,j]))
        

        self.visited.append(self.current_state)
        self.current_steps += 1
        is_terminal = int(self.current_state.terminal)
        is_done = is_terminal
        next_obs = s_p_a if self.state_type != 'image' else self._map2screen()
        return next_obs,reward,is_done,{} 

    def reset_to_state(self,state):
        for s in self.accessible_states:
            if np.all(s.coords == state):
                self.current_state = s
        self.visited = []
        self.current_steps = 0
        if self.randomize_goals:
            self._pick_goals()
        next_obs = self.current_state.coords if self.state_type != 'image' else self._map2screen()
        return next_obs
    
    def _pick_goal(self):
        current_goal = np.random.choice(self.goal_states,1)[0]
        i_current,j_current = current_goal.coords
        for goal in self.goal_states:
            i,j = goal.coords
            if not (i_current == i and j_current == j):
                self.current_map[i][j] = DiscreteState(**self.reward_mapping[self.default_state],coords=np.array([i,j]))

    def reset(self):
        self.current_map = copy.deepcopy(self.raw_map)
        self.current_state = np.random.choice(self.initial_states,1)[0]
        if self.randomize_goals:
            self._pick_goal()
        next_obs = self.current_state.coords if self.state_type != 'image' else self._map2screen()
        self.visited = []
        self.current_steps = 0
        return next_obs



class DiscreteState(object):
    def __init__(self,reward_pdf,stochastic,terminal,accessible,color,coords,initial,collectible):
        """
        A discrete representation of the state S_t.

        Args:
            reward_pdf: lambda function returning a reward ( e.g. |N(0,1)| on [0,+inf)).
            reward: Realization of the real random variable R_t.
            terminal: True if end state, False otherwise.
            color: Rgb color code (array of length 3).
            accessible: True if not a wall, False otherwise.
            coords: [x,y] position of the state.
            initial: True if one of the starting states, False otherwise.
            collectible: If true, reward becomes 0 after the first time the agent visits the state ( e.g. treasure gets taken).
        """
        self.reward_pdf = reward_pdf
        self.reward = None if stochastic else self.reward_pdf()
        self.terminal = terminal
        self.color = color
        self.accessible = accessible
        self.coords = coords
        self.initial = initial
        self.collectible = collectible
    def get_reward(self):
        return self.reward if self.reward else self.reward_pdf()

def navigation_alphabet():
    return {
            '#':{'reward_pdf':lambda :0,'terminal':False,'accessible':False,'color':[0,0,0],'stochastic':False,'initial':False,'collectible':False},
            ' ':{'reward_pdf':lambda :-1,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':False,'collectible':False},
            'S':{'reward_pdf':lambda :-1,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':True,'collectible':False},
            '0':{'reward_pdf':lambda :1,'terminal':True,'accessible':True,'color':[50,50,255],'stochastic':False,'initial':False,'collectible':False},
            '.agent':{'color':[50,205,50]}
            }

def noisy_navigation_alphabet():
    return {
            '#':{'reward_pdf':lambda :0,'terminal':False,'accessible':False,'color':[0,0,0],'stochastic':False,'initial':False},
            ' ':{'reward_pdf':lambda :np.random.normal(loc=-1,scale=0.05,size=1).item(),'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':False},
            'S':{'reward_pdf':lambda :np.random.normal(loc=-1,scale=0.05,size=1).item(),'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':True},
            '0':{'reward_pdf':lambda :10,'terminal':True,'accessible':True,'color':[50,50,255],'stochastic':False,'initial':False},
            '.agent':{'color':[50,205,50]}
            }

if __name__ == "__main__":
    env = PixelWorld(navigation_alphabet(),"####\n#S #\n## #\n####",from_string=True,actions="2d_discrete")
    print(env.current_state.coords)
    env.step(1)
    print(env.current_state.coords)
    env.reset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
