import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
import random
import copy
import matplotlib.pyplot as plt 

from model import seekndestroy
from environment import RacecarEnv

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

class genetic_algo(object):

    def __init__(self, processors, max_step=250, num_turns=5):
        self.max_step = max_step
        self.processors = processors
        self.num_turns = num_turns

    def init_weights(self, m):
        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            m.to(device)
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)
            

    def return_random_agents(self, num_agents):
        agents = []
        for _ in range(num_agents):

            agent = seekndestroy()

            for param in agent.parameters():
                param.requires_grad = False

            self.init_weights(agent)
            agents.append(agent)
            
            if torch.cuda.is_available():
                agents.cuda()

        return agents

    def step(self, agent, runs, env, seeds):

        agent.eval()
        rs = []

        for run in range(runs):

            observation = env.reset(seeds[run])
            r = 0
            s = 0

            for _ in range(self.max_step):
                inp = torch.tensor(observation).type('torch.FloatTensor')
                inp.cuda()
                mu = agent(inp)
                mu = mu.detach().numpy()
                action = mu

                action[0][1] = action[0][1] * np.pi / 4

                new_observation, reward, done, info = env.step(action)

                r = r + reward
                s = s + 1
                observation = new_observation

                if(done):
                    break

            if reward == -99:
                r += np.sqrt((env.goal[0] - env.sim.car[0])**2 + (env.goal[1] - env.sim.car[1])**2) * 10 * 3

            if not done:
                r += np.sqrt((env.goal[0] - env.sim.car[0])**2 + (env.goal[1] - env.sim.car[1])**2) * 10 * 3

            rs.append(r)

        return sum(rs) / runs

    def run_agents_n_times(self, agents, runs):

        pool = mp.Pool(processes=24)
        env = RacecarEnv(self.num_turns)
        seeds = []
        random.seed()
        for i in range(runs):
            seeds.append(random.randint(1, 10000))

        results = [pool.apply_async(self.step, args=(x, runs, env, seeds)) for x in agents]
        reward_agents = [-p.get() for p in results]
        pool.close()

        return reward_agents

    def crossover(self, father, mother, Num_Crossover):

        child_1_agent = copy.deepcopy(father)
        child_2_agent = copy.deepcopy(mother)

        cross_idx = np.random.randint(sum(p.numel() for p in father.parameters()), size=Num_Crossover)

        cnt = 0
        switch_flag = False

        father_param = list(father.parameters())

        for i, layer in enumerate(mother.parameters()):
            for j, p in enumerate(layer.flatten()):
                if cnt in cross_idx:
                    switch_flag = not switch_flag
                if switch_flag:
                    list(child_1_agent.parameters())[i].flatten()[j] = p
                    list(child_2_agent.parameters())[i].flatten()[j] = father_param[i].flatten()[j]
                cnt += 1

        return child_1_agent, child_2_agent

    def mutate(self, agent, Mutation_Power):

        child_agent = copy.deepcopy(agent)
        # mutation_power = 0.4 # 0.02 hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
        mutation_occ = 1

        for param in child_agent.parameters():

            if(len(param.shape) == 2): # weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        p = random.random()
                        if p <= mutation_occ:
                            param[i0][i1] += Mutation_Power * np.random.randn()

            elif(len(param.shape) == 1): # biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    p = random.random()
                    if p <= mutation_occ:
                        param[i0] += Mutation_Power * np.random.randn()

            return child_agent

    def return_children(self, agents, sorted_parent_indexes, elite_index, Num_Crossover, Mutation_Power):

        children_agents = []

        #first take selected parents from sorted_parent_indexes and generate N-1 children

        while len(children_agents) < 90:

            # Picking random mother and father

            father = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
            mother = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]

            child_1, child_2 = self.crossover(agents[father], agents[mother], Num_Crossover)
            child_1 = self.mutate(child_1, Mutation_Power)
            child_2 = self.mutate(child_2, Mutation_Power)

            children_agents.extend([child_1, child_2])

        for i in range(9):
            mutant = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
            mutant = self.mutate(agents[mutant], Mutation_Power)

            children_agents.append(mutant)

        #now add one elite
        elite_child, top_score = self.add_elite(agents, sorted_parent_indexes, elite_index)
        children_agents.append(elite_child)
        elite_index = len(children_agents) - 1 # it is the last one

        return children_agents, elite_index, top_score

    def add_elite(self, agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):

        candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

        if(elite_index is not None):
            candidate_elite_index = np.append(candidate_elite_index, [elite_index])

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

    def train(self, num_agents, generations, top_limit, file, Num_Crossover, Mutation_Power):

        agents = self.return_random_agents(num_agents)

        elite_index = None
        
        Fitness = []
        
        for generation in range(generations):
            # return rewards of agents
            rewards = self.run_agents_n_times(agents, 3) # return average of 3 runs

            sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

            print("\n\n", sorted_parent_indexes)

            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(rewards[best_parent])

            Fitness.append(min(top_rewards))
            
            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ", np.mean(top_rewards[:5]))
            print("Top ", top_limit, " scores", sorted_parent_indexes)
            print("Rewards for top: ", top_rewards)

            # setup an empty list for containing children agents
            children_agents, elite_index, top_score = self.return_children(agents, sorted_parent_indexes, elite_index, Num_Crossover, Mutation_Power)

            # kill all agents, and replace them with their children
            agents = children_agents

            # Saving weights
            if generation % 10 == 0:
                torch.save(agents[elite_index].state_dict(), 'models/' + file + '_{}'.format(generation))
                plt.plot(np.arange(len(Fitness)), Fitness, 'rD', markersize=9, label =('The Top Rewards'))
                plt.xlabel('Epochs')
                plt.ylabel('Fitness')
                plt.legend()
                plt.show()