#import logging
from gym.envs.registration import register

register(
    id='RacecarEnv-v0',
    entry_point='Racecar.envs.environment:RacecarEnv',
)
