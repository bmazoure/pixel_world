from gym.envs.registration import register

from pixel_world.env_utils import navigation_alphabet, noisy_navigation_alphabet, PixelWorld
import os 
is_windows = os.name == 'nt'
sep = '/' if not is_windows else '\\'

__all__ = ['navigation_alphabet', 'noisy_navigation_alphabet', 'PixelWorld']

register(
    id='PixelWorld-v0',
    entry_point='pixel_world.envs:PixelWorld',
    kwargs={'reward_mapping':navigation_alphabet(),'world_map':"maps"+sep+"room1_small.txt",'from_string':False}
)
register(
    id='PixelWorld-v1',
    entry_point='pixel_world.envs:PixelWorld',
    kwargs={'reward_mapping':noisy_navigation_alphabet(),'world_map':"maps"+sep+"room2_small.txt",'from_string':False}
)
register(
    id='PixelWorld-v2',
    entry_point='pixel_world.envs:PixelWorld',
    kwargs={'reward_mapping':noisy_navigation_alphabet(),'world_map':"maps"+sep+"room3_small.txt",'from_string':False}
)
register(
    id='PixelWorld-v3',
    entry_point='pixel_world.envs:PixelWorld',
    kwargs={'reward_mapping':noisy_navigation_alphabet(),'world_map':"maps"+sep+"room4_small.txt",'from_string':False}
)
register(
    id='PixelWorld-v4',
    entry_point='pixel_world.envs:PixelWorld',
    kwargs={'reward_mapping':noisy_navigation_alphabet(),'world_map':"maps"+sep+"room5_medium.txt",'from_string':False}
)
register(
    id='PixelWorld-v5',
    entry_point='pixel_world.envs:PixelWorld',
    kwargs={'reward_mapping':noisy_navigation_alphabet(),'world_map':"maps"+sep+"room5_medium_walls.txt",'from_string':False}
)
