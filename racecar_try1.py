import os
import sys
import simpy
import gym
from gym import wrappers
#import model as mod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from descartes import PolygonPatch
import shapely.geometry as sg

sys.path.append('/Users/macbookpro/Desktop/Racecar_Project')
sys.path.append('/Users/macbookpro/Desktop/Racecar_Project/Racecar')
sys.path.append('/Users/macbookpro/Desktop/Racecar_Project/Racecar/envs')

from Racecar.envs.racetrack import racetrack
from Racecar.envs.simulation import python_env
from Racecar.envs.environment import RacecarEnv

from gym import envs
import Racecar

# envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)


num_turns = 4
env = gym.make('Racecar-v0', turns = num_turns)

class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(PolicyNet, self).__init__()
        # network
#         self.hidden = nn.Linear(n_inputs, n_hidden)
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_outputs)
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.D = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.hidden(x)
#         x = self.D(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.out(x)
        return F.softmax(x, dim=1)
    
    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))


n_inputs = env.observation_space.shape[0]
n_hidden = 35
n_outputs = 2 #env.action_space.n

print('state shape:', n_inputs)
print('action shape:', n_outputs)


def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns

# training settings

num_episodes = 10
rollout_limit = 10 # max rollout length
discount_factor = 0.99 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.01 # you know this by now
val_freq = 10 # validation frequency

# Policy 
policy = PolicyNet(n_inputs, n_hidden, n_outputs, learning_rate)

def closer(dist1, dist2):
    return dist2 < dist1

# train policy network
velocity = 1
steering_angle = 0
theta = []
RP = []

try:
    training_rewards, losses = [], []
    print('start training')
    for i in range(num_episodes):
        x = []
        y = []
        rollout = []
        R = racetrack(num_turns,i)
        RacePlot, Goal = R.generate()
        P = python_env(num_turns,i)
        s = env.reset(i)
        R.generate(1)
#         print(Goal[-1])
#         RacePlot, Goal = R.generate()



# PLOT TRACK:

#         track = sg.LineString(Goal)
#         outer = track.buffer(1.5)
#         inner = outer.buffer(-0.5)
#         Track = outer - inner
        xs = [a[0] for a in Goal]
        ys = [a[1] for a in Goal]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.set_xlim(min(xs)-5,max(xs)+5)
        ax.set_ylim(min(ys)-5,max(ys)+5)
        ax.plot(xs, ys)
        ax.add_patch(PolygonPatch(RacePlot, alpha=1, zorder=2))
        print("Position:", (x,y,theta))


#         ss = s[0][:30]
#         print("\nThese are the states:\n", ss)
#         D = s[0][30:33]
#         print("\nThese are the distance, angle, sign:\n", D)
#         a = s[0][-2:]
#         print("\nThis is the action:\n", a)




        for j in range(rollout_limit):
            
            Dist,Angle,Sign = env.observe()
            print("Observation: Distance | Angle | Sign:",Dist, Angle, Sign)
            x1,y1,theta = P.kinematic(velocity, steering_angle)
            x.append(x1)
            y.append(y1)
            if closer(Angle, theta) == 1:
                steering_angle = theta

            if closer(Dist,x[j]) == 1:
                velocity = velocity+1
            else:
                velocity = velocity/2
            
#             if (j % 5) == 0:
#                 xR,yR = RacePlot.exterior.xy
#                 L1 = plt.plot(xR,yR)
#         L2 = plt.plot(x,y)
#         L3 = plt.plot(Goal[0],Goal[1])
#         plt.show()

        
            # generate rollout by iteratively evaluating the current policy on the environment
            with torch.no_grad():
                a_prob = policy(torch.from_numpy(np.atleast_2d(s)).float())
#                 print(a_prob)
            a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action
#             print(a)
            s1, r, done, _ = env.step([[a,a],[a,a]])
#             s1, r, done, _ = env.step(a_prob)
#             dist, phi, sign = env.observe()
#             print("Distance:", dist)
            rollout.append((s, a, r))
#             print(len(rollout[0]))
            s = s1
            if done: break
            
            
        L2 = ax.plot(x,y, 'r--')
        L3 = ax.plot(Goal[-1][0],Goal[-1][1], 'rX',markersize= 20)
        plt.show()    
        # prepare batch
        rollout = np.array(rollout)
#         print(np.shape(rollout[0][0]))
        states = np.vstack(rollout[:,0])
#         print(np.shape(states))
        actions = np.vstack(rollout[:,1])
#         print(np.shape(actions))
        rewards = np.array(rollout[:,2], dtype=float)
        print(rewards)
        returns = compute_returns(rewards, discount_factor)
        # policy gradient update
        policy.optimizer.zero_grad()
        a_probs = policy(torch.from_numpy(states).float()).gather(1, torch.from_numpy(actions)).view(-1)
        loss = policy.loss(a_probs, torch.from_numpy(returns).float())
        loss.backward()
        policy.optimizer.step()
        # bookkeeping
        training_rewards.append(sum(rewards))
        losses.append(loss.item())
        # print
        if (i+1) % val_freq == 0:
            # validation
            validation_rewards = []
            for _ in range(10):
                s = env.reset(i)
                reward = 0
                for _ in range(rollout_limit):
                    with torch.no_grad():
                        a = policy(torch.from_numpy(np.atleast_2d(s)).float()).argmax().item()
                    s, r, done, _ = env.step([[a,a],[a,a]])
                    reward += r
                    if done: break
                validation_rewards.append(reward)
            print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, np.mean(training_rewards[-val_freq:]), np.mean(validation_rewards), np.mean(losses[-val_freq:])))
    print('done')
except KeyboardInterrupt:
    print('interrupt')    

A = [[1,2,3,4],
    [5,6,7,8]]
# torch.tensor(X_train).float()
for i_episode in range(10):
    s = env.reset(5)
    print(torch.tensor(s).float())
    print(torch.tensor(s).float().argmax().item())
    for t in range(10):
#         action = A #env.observation_space.sample()
        a = policy(torch.from_numpy(np.atleast_2d(s)).float()).argmax().item()
#         a = policy(torch.from_numpy(np.atleast_2d(s))).argmax().item()
        print(a)
        s, r, done, _ = env.step(a)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
# env.step()
# env.reset()

def get_user_input():
    
    Coord = input("Input relative coordinates: ")
    Dist = input("Input distances to the nearest objects: ")
    
    params = [Coord, Dist]
    if all(str(i).isdigit() for i in params):  # Check input is valid
        params = [int(x) for x in params]
    else:
        print(
            "Could not parse input. The simulation will use default values:",
            "\n1 cashier, 1 server, 1 usher.",
        )
        params = [1, 1]
    return params

print("Nigga")