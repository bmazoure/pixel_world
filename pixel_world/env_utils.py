import numpy as np
import torch
import matplotlib.pyplot as plt

class DiscreteState(object):
    def __init__(self,reward_pdf,stochastic,terminal,accessible,color,coords,initial):
        self.reward_pdf = reward_pdf # e.g. |N(0,1)| on [0,+inf)
        self.reward = None if stochastic else self.reward_pdf()
        self.terminal = terminal
        self.color = color
        self.accessible = accessible
        self.coords = coords
        self.initial = initial
    def get_reward(self):
        return self.reward if self.reward else self.reward_pdf()

def navigation_alphabet():
    return {
            '#':{'reward_pdf':lambda :0,'terminal':False,'accessible':False,'color':[0,0,0],'stochastic':False,'initial':False},
            ' ':{'reward_pdf':lambda :-1,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':False},
            'S':{'reward_pdf':lambda :-1,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':True},
            '0':{'reward_pdf':lambda :0,'terminal':True,'accessible':True,'color':[50,50,255],'stochastic':False,'initial':False},
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

def plot_screen(env,ax):
    ax.imshow(env._map2screen())
    plt.show()
