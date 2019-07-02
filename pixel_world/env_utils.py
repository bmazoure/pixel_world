import numpy as np
import torch
import matplotlib.pyplot as plt

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
