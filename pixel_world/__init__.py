from gym.envs.registration import register

from .env_utils import navigation_alphabet
import os 
is_windows = os.name == 'nt'
dir_path = os.path.dirname(os.path.realpath(__file__))
project_root = '/'.join(dir_path.split('/')[:-1]) if not is_windows else os.getcwd()+"\\pixel_world"

sep = '/' if not is_windows else '\\'

register(
    id='PixelWorld-v0',
    entry_point='pixel_world.envs:PixelWorld',
    kwargs={'reward_mapping':navigation_alphabet(),'world_map':project_root+sep+"maps"+sep+"room1_small.txt"}
)