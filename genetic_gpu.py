import torch
import torch.nn as nn
import numpy as np
import random
import copy
import matplotlib.pyplot as plt 

# from model import seekndestroy
from New_model import seekndestroy
from environment import RacecarEnv
from environment import closer

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

class genetic_algo(object):

    def __init__(self, processors, num_turns, max_step=250):
        self.max_step = max_step
        self.processors = processors
        self.num_turns = num_turns

    def init_weights(self, m):
        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            m.to(device)
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)

    def update_weights(self, m, checkpoint):
      if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
        m.to(device)
        # m.get_device()
        m.load_state_dict(checkpoint['state_dict'])
          # Update the parameter.
          # New_agents[name].copy_(param)
             

    def return_updated_weights(self, num_agents,checkpoint):

        agents = []
        for _ in range(num_agents):

            agent = seekndestroy().cuda()

            for param in agent.parameters():
                param.requires_grad = False

            self.update_weights(agent, checkpoint)
            agents.append(agent)

        return agents

    def return_random_agents(self, num_agents):

        agents = []
        for _ in range(num_agents):

            agent = seekndestroy().cuda()
                   
            for param in agent.parameters():
                param.requires_grad = False

            self.init_weights(agent)
            agents.append(agent)
           
        return agents

    def step(self, agent, runs, env, seeds):

        agent.eval()
        rs = []
        Goal_Counter = 0

        # print("NEXT AGENT")

        for run in range(runs):

            # print("Run number:", run)

            observation = env.reset(seeds[run])
            r = 0
            s = 0
            

            for _ in range(self.max_step):
                inp = torch.tensor(observation).type('torch.FloatTensor').cuda()
                                
                # print(torch.cuda.device_count())
                if torch.cuda.device_count() > 1:
                  print("Let's use", torch.cuda.device_count(), "GPUs!")
       
                agent.to(device)
                
                mu = agent(inp)
                mu = mu.cpu().detach().numpy()
                action = mu.reshape(1,2)
                

                action[0][1] = action[0][1] * np.pi / 4

                new_observation, reward, done, info = env.step(action)
                
                

                if(done):
                    break

                
                
                r = r + reward
                s = s + 1
                
                old_parameters = np.array(np.reshape(observation, (1, 35)))
                
                print(old_parameters)
                old_Lidar = old_parameters[:,1:30]
                print(sum(old_Lidar))
                old_DIST = old_parameters[:,-5]
                # print(old_DIST)
                old_ANGLE = old_parameters[:,-4]
                
                observation = new_observation
                
                
                parameters = np.array(np.reshape(observation, (1, 35)))
                
                
                new_Lidar = parameters[:,1:30]
                print(new_Lidar)
                print(sum(new_Lidar))
                new_DIST = parameters[:,-5]
                # print(new_DIST)
                new_ANGLE = parameters[:,-4]

            if reward == 0:
                  Goal_Counter += 1        
                    
                    
        ## Adding a penalty for the physical restraints of the system, the car cannot turn more than a certain amount. The exact angle yet to be determined 

            if new_ANGLE > np.pi/2:   
                r += 100
                print("Recieved extra angle penalty")
                
                
######### 15 is an estimate for when half of the lidar measurements (30 total) are 0.5 meaning starting to head towards a wall. You want to minimize this value because the lidar scan returns the distance of a dectection of an obstacle, then its subtracted from the length of the track (1-scan) which is essentially then the amount the car deviated from the track. Penalized if facing a wall too much:

            if sum(new_Lidar) > 15:   
                r += sum(new_Lidar)*10
            
            
            ## Here the car is penalized if facing a wall too much and heading towards it:  
            
            if ((sum(new_Lidar) > 15) and (sum(old_Lidar) < sum(new_Lidar))):
                r += abs(sum(new_Lidar))*3 
            
            if closer(old_DIST,new_DIST) == 0:
                r += np.abs((old_DIST-new_DIST))*10 * 3
                # print("Recieved extra distance penalty")    


            if reward == -99:
                r += np.sqrt((env.goal[0] - env.sim.car[0])**2 + (env.goal[1] - env.sim.car[1])**2) * 10 * 3

            if not done:
                r += np.sqrt((env.goal[0] - env.sim.car[0])**2 + (env.goal[1] - env.sim.car[1])**2) * 10 * 3

            rs.append(r)
        GOALS_TOTAL.append(np.copy(Goal_Counter))
        # print(Goal_Counter)
        # print(sum(GOALS_TOTAL))

        return sum(rs) / runs #, G_Goals

    def run_agents_n_times(self, agents, runs):

        env = RacecarEnv(self.num_turns)
        seeds = []
        random.seed()
        for i in range(runs):
            seeds.append(random.randint(1, 10000))

   
        results = [self.step(x,runs,env,seeds) for x in agents]
        # print(np.shape(results))

        # G_Goals = np.array(results)[:,1]
        # # print(np.shape(G_Goals))
        # results = np.array(results)[:,0]



        reward_agents = []
        for i in range(len(results)):
          reward_agents.append(-results[i])
        

        return reward_agents  #, G_Goals

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
            # print(father)
            mother = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
            # print(mother)
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
        GOALS_HIT = []
        # global Goal_Counter 
        # Goal_Counter = 0
        global GOALS_TOTAL
        GOALS_TOTAL = []
        
        for generation in range(generations):
            # return rewards of agents

            
            rewards = self.run_agents_n_times(agents, 3) # return average of 3 runs
            # print(rewards)
            # G_Goals = np.array(rewards)[:,1]
            # rewards = np.array(rewards)[:,0]
            
            if sum(np.shape(rewards))!= len(rewards):
              rewards = np.array(rewards).ravel()

            # print(rewards)
            # print(np.shape(rewards))

            sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

            print("\n\n", sorted_parent_indexes)

            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(np.array(rewards)[best_parent])

            Fitness.append(min(top_rewards))
            # GOALS_HIT.append(Goal_Counter)
            GOALS_HIT.append(sum(GOALS_TOTAL))
            
            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ", np.mean(top_rewards[:5]))
            print("Top ", top_limit, " scores", sorted_parent_indexes)
            print("Rewards for top: ", top_rewards)

            # setup an empty list for containing children agents
            children_agents, elite_index, top_score = self.return_children(agents, sorted_parent_indexes, elite_index, Num_Crossover, Mutation_Power)

            # kill all agents, and replace them with their children
            agents = children_agents
      
            # Saving weights
            # if generation % 10 == 0:
            #   plt.bar(np.arange(len(GOALS_HIT)),GOALS_HIT)
            #   plt.xlabel('Generations')
            #   plt.ylabel('Goals Hit')
            #   plt.show()
              
            #   # Curriculum learning, update the children agents to start from the best saved parent agents
            #   PATH = 'models/' + file + '_{}'.format(generation)
            #   torch.save(agents[elite_index].state_dict(), PATH)
            #   agents = self.return_updated_weights(agents, PATH)           

              # optimizer = TheOptimizerClass(*args, **kwargs)
              # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
              # checkpoint = torch.load(PATH)

            if generation == generations-1:
              PATH = 'models/' + 'turn'+ '_{}'.format(self.num_turns)
              torch.save(agents[elite_index].state_dict(), PATH)
              # Save the final fitness plot
              f = plt.figure(1)  
              plt.plot(np.arange(len(Fitness)), Fitness, '-o', markersize=9, label =('The Top Rewards'))
              plt.xlabel('Epochs')
              plt.ylabel('Fitness')
              plt.legend()
              figname = 'Fitness' + '_{}'.format(self.num_turns)
              f.savefig(figname +'.pdf')
              g = plt.figure(2)
              plt.bar(np.arange(len(GOALS_HIT)),GOALS_HIT)
              plt.xlabel('Generations')
              plt.ylabel('Max Goals Hit')
              figname2 = 'Goals' + '_{}'.format(self.num_turns)
              g.savefig(figname2 +'.pdf')
              plt.show()

    def Curriculum_train(self, num_agents, generations, top_limit, file, Num_Crossover, Mutation_Power):

        LOAD_PATH = 'models/' + file
        checkpoint = torch.load(LOAD_PATH)
        agents = self.return_updated_weights(num_agents,checkpoint)
        # agents.load_state_dict(checkpoint['state_dict'])
        
        
        elite_index = None
        
        Fitness = []
        GOALS_HIT = []
        # global Goal_Counter 
        # Goal_Counter = 0
        global GOALS_TOTAL
        GOALS_TOTAL = []
        
        for generation in range(generations):
            # return rewards of agents

            
            rewards = self.run_agents_n_times(agents, 3) # return average of 3 runs
            # print(rewards)
            if sum(np.shape(rewards))!= len(rewards):
              rewards = np.array(rewards).ravel()

            sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

            print("\n\n", sorted_parent_indexes)

            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(np.array(rewards)[best_parent])

            Fitness.append(min(top_rewards))
            # GOALS_HIT.append(Goal_Counter)
            GOALS_HIT.append(sum(GOALS_TOTAL))
            
            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ", np.mean(top_rewards[:5]))
            print("Top ", top_limit, " scores", sorted_parent_indexes)
            print("Rewards for top: ", top_rewards)

            # setup an empty list for containing children agents
            children_agents, elite_index, top_score = self.return_children(agents, sorted_parent_indexes, elite_index, Num_Crossover, Mutation_Power)

            # kill all agents, and replace them with their children
            agents = children_agents
      
            # Saving weights
            # if generation % 10 == 0:
              
            #   # Curriculum learning, update the children agents to start from the best saved parent agents
            #   PATH = 'models/' + file + '_{}'.format(generation)
            #   torch.save(agents[elite_index].state_dict(), PATH)
            #   agents = self.return_updated_weights(agents, PATH)           

              # optimizer = TheOptimizerClass(*args, **kwargs)
              # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
              # checkpoint = torch.load(PATH)

            if generation == generations-1:
              PATH = 'models/' + 'turn'+ '_{}'.format(self.num_turns)
              torch.save(agents[elite_index].state_dict(), PATH)
              # Save the final fitness plot
              f = plt.figure(1)  
              plt.plot(np.arange(len(Fitness)), Fitness, '-o', markersize=9, label =('The Top Rewards'))
              plt.xlabel('Epochs')
              plt.ylabel('Fitness')
              plt.legend()
              figname = 'Fitness' + '_{}'.format(self.num_turns)
              f.savefig(figname +'.pdf')
              g = plt.figure(2)
              plt.bar(np.arange(len(GOALS_HIT)),GOALS_HIT)
              plt.xlabel('Generations')
              plt.ylabel('Max Goals Hit')
              figname2 = 'Goals' + '_{}'.format(self.num_turns)
              g.savefig(figname2 +'.pdf')
              plt.show()