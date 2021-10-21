#import logging
from gym.envs.registration import register

register(
    id='Racecar-v0',
    entry_point='Racecar.envs.environment:RacecarEnv',
)
