import os
import torch
import torch.nn as nn

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import numpy as np
import multiprocessing as mp
import random
import copy
import matplotlib.pyplot as plt
import pandas as pd        
import cupy as cp


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from gym import wrappers

cuda = torch.device('cuda') 

from model import seekndestroy 
from environment import RacecarEnv

class genetic_algo(object):
    
    def __init__(self, processors, max_step=250, num_turns=5):
        self.max_step = max_step
        self.processors = processors
        self.num_turns = num_turns

    #used by return_random_agents
    def init_weights(self, m):
        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)

    #used by train
    def return_random_agents(self, num_agents):
        agents = []
        for _ in range(num_agents):

            agent = seekndestroy() #this returns a NN architecture ! 
            agent = agent.cuda()

            for param in agent.parameters():
                param.requires_grad = False # no grad because genetic / no backprop

            self.init_weights(agent) 
            agents.append(agent)

        return agents

    # used by run_agents_n_times
    def step(self, agent, runs, env, seeds):                                   
        agent.eval() #this takes us to evaluation/testing mode. Kind of a switch
        rs = [] #empty array

        for run in range(runs):
            observation = env.reset(seeds[run]) # this returns the state, and resets/cleans ..PRINT IT AT SOME POINT 
            #seeds???
            
            r = 0
            s = 0

            for _ in range(self.max_step): # 250 steps 
                inp = torch.Tensor(observation).type('torch.cuda.FloatTensor')
                mu = agent(inp) # we input the observation into our agent. observation == input layer 
                
                mu = mu.detach().cpu().numpy()

                
                #mu = cp.fromDlpack(to_dlpack(mu)) # don't matter to gradient, don't record operations on this tensor
                #mu = cp.asarray(mu)
                
                action = mu #action is output of the system, in numpy format, with no recording (detached

                # action is matrix of 2 * something
                action[0][1] = action[0][1] * (cp.pi/4)  # multiply by pi/4 . pi/4 = 45 degrees. something with direction ... 

                new_observation, reward, done, info = env.step(action)
                # we take a step, new_obs is the new state of the system
                # reward is the reward for taking that step
                # How the F do you define rewards ??? in plm 
                # done tells us when to stop

                r = r + reward #sums of rewards for this run !
                s = s + 1
                observation = new_observation # state = new state

                if(done):
                    break

            if reward == -99:
                r += cp.sqrt((env.goal[0] - env.sim.car[0])**2 + (env.goal[1] - env.sim.car[1])**2) * 10 * 3 # ???
                # -99 is reward when hitting wall , maybe, bad anyways 
                
                # times 30 is some arbitrary scaling factor.  

            if not done:
                r += cp.sqrt((env.goal[0] - env.sim.car[0])**2 + (env.goal[1] - env.sim.car[1])**2) * 10 * 3 # you do same as rew == -99


            # reward is actually something you wanna minimize!!! Hit goal faster . 


            rs.append(r) #append the sum of rewards for this run, to one for all runs

        return sum(rs) / runs # return average reward for all the runs
    

    # used by add_elite and train
    def run_agents_n_times(self, agents, runs):

        #pool = mp.Pool(processes=24) # multithreading programming, max 24 processes
        env = RacecarEnv(self.num_turns)  
        seeds = []
        random.seed() #initialize the random number generator
        for i in range(runs):
            seeds.append(random.randint(1, 10000)) #wtf are seeds??

        #results = [pool.apply_async(self.step, args=(x, runs, env, seeds)) for x in agents]
        
        results = []
        for x in agents:
            result = self.step(x, runs, env, seeds) 
            results.append(result)
        
        #reward_agents = [-p.get() for p in results] # why minus ???? 

        reward_agents = [-p for p in results]
        #pool.close() #close multithread

        return reward_agents

    # used by return_children
    def crossover(self, father, mother, crossover_num = 2):

        child_1_agent = copy.deepcopy(father)
        child_2_agent = copy.deepcopy(mother)
        # Shallow copy takes the copy. Deep copy creates a new memory place to store the values

        cross_idx = cp.random.randint(sum(p.numel() for p in father.parameters()), size=crossover_num)  #   ?????? 
        # numel returns the total number or elements in the tensor (dim1*dim2*....)
        # why randint(--- , 2) ??
        # generate new parameters . 

        cnt = 0
        switch_flag = False

        father_param = list(father.parameters())
        # Mom and pop are randomly selected from best_index_parents array

        for i, layer in enumerate(mother.parameters()):  # Two-index count in enumerate() -> index + element
            for j, p in enumerate(layer.flatten()): # We look into each layer (element in mom.param)
                if cnt in cross_idx:
                    switch_flag = not switch_flag
                if switch_flag:
                    list(child_1_agent.parameters())[i].flatten()[j] = p       # passes parameters(which, idk?) from mom to kid
                    list(child_2_agent.parameters())[i].flatten()[j] = father_param[i].flatten()[j] # passes parameters(which, idk?) from dad to kid
                cnt += 1
                # cross_idx is either 1 or 2 (maybe 0). 
        
        # Loops through all weights (params) in mom and dad. Some get passed (when flad it true), which is aprox. half. Some don't.
        # One child gets dad one gets mom. 


        return child_1_agent, child_2_agent

    # used by return_children
    def mutate(self, agent):

        child_agent = copy.deepcopy(agent)
        mutation_power = 0.02 # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
        mutation_occ = 1

        for param in child_agent.parameters():

            if(len(param.shape) == 2): # weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        p = random.random()
                        if p <= mutation_occ:
                            param[i0][i1] += mutation_power * cp.random.randn()
                        # if this random dice < 1, we do a mutation with some other random number * 0.02

            elif(len(param.shape) == 1): # biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    p = random.random()
                    if p <= mutation_occ:
                        param[i0] += mutation_power * cp.random.randn()

            return child_agent
            # This is just a random mutation at a random position. Pretty random. 


    # used by train
    def return_children(self, agents, sorted_parent_indexes, elite_index):

        children_agents = []

        #first take selected parents from sorted_parent_indexes and generate N-1 children

        while len(children_agents) < 90: #here we add 90 kids

            # Picking random mother and father

            father = sorted_parent_indexes[cp.random.randint(len(sorted_parent_indexes))]
            mother = sorted_parent_indexes[cp.random.randint(len(sorted_parent_indexes))]

            child_1, child_2 = self.crossover(agents[father], agents[mother])
            child_1 = self.mutate(child_1)
            child_2 = self.mutate(child_2)

            children_agents.extend([child_1, child_2])

        for i in range(9): # here we add 9 mutated parents
            mutant = sorted_parent_indexes[cp.random.randint(len(sorted_parent_indexes))]
            mutant = self.mutate(agents[mutant])

            children_agents.append(mutant)
        # Try all combos: children crossover + parent with mutation . To get even more randomness ! yey 

        #now add one elite. Hercule 
        elite_child, top_score = self.add_elite(agents, sorted_parent_indexes, elite_index)
        children_agents.append(elite_child)
        elite_index = len(children_agents) - 1 # it is the last one

        return children_agents, elite_index, top_score



        ### semn de carte @@@ 


    # used by return_children
    def add_elite(self, agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):

        candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

        if(elite_index is not None):
            candidate_elite_index = cp.append(candidate_elite_index, [elite_index])

        top_score = None
        top_elite_index = None

        test_agents = [agents[i] for i in candidate_elite_index]
        scores = self.run_agents_n_times(test_agents, runs=5)

        for n, i in enumerate(candidate_elite_index):
            score = scores[n]
            print("Score for elite i ", i, " is ", score)

            if(top_score is None):
                top_score = score
                top_elite_index = i
            elif(score > top_score):
                top_score = score
                top_elite_index = i

        print("Elite selected with index ", top_elite_index, " and score", top_score)

        child_agent = copy.deepcopy(agents[top_elite_index])

        return child_agent, top_score

    def train(self, num_agents, generations, top_limit, file):

        agents = self.return_random_agents(num_agents)

        elite_index = None

        for generation in range(generations):
            # return rewards of agents
            rewards = self.run_agents_n_times(agents, 3) # return average of 3 runs

            # Top limis is the number of limits in the elite. How many agents reproduce. 

            sorted_parent_indexes = cp.argsort(rewards)[ : :-1][ :top_limit] # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
            # returns top rewards and puts them in sorted_parents_indexes    


            print("\n\n", sorted_parent_indexes)

            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(rewards[best_parent])

            print("Generation ", generation, " | Mean rewards: ", cp.mean(rewards), " | Mean of top 5: ", cp.mean(top_rewards[:5]))
            print("Top ", top_limit, " scores", sorted_parent_indexes)
            print("Rewards for top: ", top_rewards)

            # setup an empty list for containing children agents
            children_agents, elite_index, top_score = self.return_children(agents, sorted_parent_indexes, elite_index)

            # kill all agents, and replace them with their children
            agents = children_agents

            # Saving weights
            if generation % 10 == 0:
                #if not os.path.exists('/models/' + file + '_{}'.format(generation)):
                    #os.makedirs('models/' + file + '_{}'.format(generation), exist_ok=True)
                #path = "Users/macbookpro/Desktop/Racecar_Project/Racecar/envs/models/"

                torch.save(agents[elite_index].state_dict(), 'Racecar/envs/models/' + file + '_{}'.format(generation))


if __name__ == "__main__":
    
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))

    algo = genetic_algo(2)  
    algo.train(10, 30,  10, 'test')


