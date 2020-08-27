import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tetris.grouped_tetris_simplest import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # save transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 32)
        #self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(32,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

### CONFIGURATIONS ###
batch_size = 64
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
target_update = 15 
#n_actions = 3

policy_net = DQN().float()
target_net = DQN().float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0
game_rewards = []

def select_action(states):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    # epsilon greedy strategy
    if sample > eps_threshold:
        with torch.no_grad():
            vals = [policy_net(state) for state in states]
            return torch.tensor(np.argmax(np.array(vals))) # state with largest expected reward
    else:
        return torch.tensor([random.randrange(len(states))], dtype=torch.long) # random action

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions)) # transpose batch

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None], dim=0)
    state_batch = torch.stack(batch.state, dim=0)
    #action_batch = torch.stack(batch.action, dim=0)
    reward_batch = torch.stack(batch.reward, dim=0)

    # compute V(s_t):
    state_values = policy_net(state_batch)

    # compute V(s_{t+1}) for all next states
    # note that expected values of actions for non_final_next_states are computed based on 'older' target_net
    next_state_values = torch.zeros(batch_size, 1)
    # merge such that expected state value (non_final) or zero (final)
    next_state_values[non_final_mask, :] = target_net(non_final_next_states).detach()
    # compute expected Q values
    expected_state_values = (next_state_values * gamma) + reward_batch

    # compute Huber loss
    loss = F.smooth_l1_loss(state_values, expected_state_values)

    # optimize model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_rewards():
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(game_rewards, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('episode')
    plt.ylabel('discounted reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

### main training loop ###
num_games_train = 1000

scores_train = []
lines_train = []
pieces_train = []
actions_train = []

for i_game in range(num_games_train+1):
    show = False
    if i_game % 100 == 0:
        print("game", i_game)
    #print(eps_end + (eps_start - eps_end) * \
    #                math.exp(-1. * steps_done / eps_decay))
    #show = True
    # initialize environment and state
    env = TetrisApp()
    state = torch.tensor(env.state, dtype=torch.float)
    score = 0
    #for t in count():
    t = 0
    T = 10
    if i_game > 100:
        T = 15
    if i_game > 200:
        T = 20
    while t < T:
        # select and perform action
        states = [torch.tensor(s, dtype=torch.float) for s in env.get_next_states()]
        if t == 0:
            action = torch.tensor([(i_game%5)*2], dtype=torch.long) # random first action (in even place, nice!)
        else:
            action = select_action(states)
        reward, done = env.step_grouped(action.item(), show)
        if reward >= 20: # at least cleared one line
            T += 10
        reward = torch.tensor([reward])
        score += gamma*reward

        # observe next state
        if not done:
            next_state = torch.tensor(env.state, dtype=torch.float)
        else:
            next_state = None
        # store transition in memory
        # including clean up of replay memory: if at least cleared one line, overrepresentation!
        memory.push(state, action, next_state, reward)
        if reward >= 20:
            memory.push(state, action, next_state, reward)
            memory.push(state, action, next_state, reward)
            memory.push(state, action, next_state, reward)

        state = next_state
        # perform one step of optimization (on target network)
        if i_game > 1:
            optimize_model()
        if done:
        #    game_rewards.append(score)
        #    plot_rewards()
            break
        t += 1

    game_rewards.append(score)
    if i_game % 100 == 0:
        plot_rewards()
        torch.save(policy_net.state_dict(), "block_drop_eps_{}_alt.pt".format(i_game))

    # update target network, copying all weights and biases in DQN
    if i_game % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # performance metrics
    i_score_train, i_lines_train, i_pieces_train, i_actions_train = env.get_perf()
    scores_train.append(i_score_train)
    lines_train.append(i_lines_train)
    pieces_train.append(i_pieces_train)
    actions_train.append(i_actions_train)

print('Complete (training)')

avg_scores_train = sum(scores_train)/len(scores_train)
avg_lines_train = sum(lines_train)/len(lines_train)
avg_pieces_train = sum(pieces_train)/len(pieces_train)
avg_actions_train = sum(actions_train)/len(actions_train)

print("RESULTS - TRAINING\n")
print("score:", avg_scores_train)
print("lines:", avg_lines_train)
print("pieces:", avg_pieces_train)
print("actions:", avg_actions_train)

env.quit()
plt.ioff()
plt.show()
