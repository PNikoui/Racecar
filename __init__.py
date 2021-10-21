import logging 
from gym.envs.registration import register  

logger = logging.getLogger(__name__)  

register(   
    id='racetrack-v0', 
    entry_point='gym_Racecar.envs:racetrack',    
#     timestep_limit=1000,
#     turns = 5,
    reward_threshold=8.0,
    nondeterministic = True, 
)

register(     
    id='simulation-v0',     
    entry_point='gym_Racecar.envs:python_env',  
#     timestep_limit=1000,
#     turns = 5,
    reward_threshold=10.0,    
    nondeterministic = True, 
)

register(     
    id='RacecarEnv-v0',     
    entry_point='gym_Racecar.envs:RacecarEnv',    
#     timestep_limit=1000,
#     turns = 5,
    reward_threshold=1.0,    
    nondeterministic = True,
)


