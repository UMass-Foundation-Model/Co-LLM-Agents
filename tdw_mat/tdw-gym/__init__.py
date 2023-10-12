from gym.envs.registration import register
from .tdw_gym import *

register(
    id='transport_challenge_MA',
    entry_point='tdw_gym:TDW'
)