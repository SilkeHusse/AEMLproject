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

from tetris.tetris_simplestgame import *

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
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 25)
        #self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(25,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

### CONFIGURATIONS ###
batch_size = 64
gamma = 0.99
eps_start = 0.5
eps_end = 0.05
eps_decay = 1000#0
target_update = 5 # update 200 times
n_actions = 3

policy_net = DQN().float()
target_net = DQN().float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0
game_rewards = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    # epsilon greedy strategy
    if sample > eps_threshold:
        with torch.no_grad():
            q_vals = policy_net(state)
            return q_vals.max(0)[1].view(1) # action with largest expected reward
    else:
        return torch.tensor([random.randrange(n_actions)], dtype=torch.long) # random action

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions)) # transpose batch

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None], dim=0)
    state_batch = torch.stack(batch.state, dim=0)
    action_batch = torch.stack(batch.action, dim=0)
    reward_batch = torch.stack(batch.reward, dim=0)

    # compute Q(s_t, a):
    # select respective actions for each batch state which would have been taken according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # compute V(s_{t+1}) for all next states
    # note that expected values of actions for non_final_next_states are computed based on 'older' target_net
    next_state_values = torch.zeros(batch_size, 1)
    # merge such that expected state value (non_final) or zero (final)
    next_state_values[non_final_mask, 0] = target_net(non_final_next_states).max(1)[0].detach()
    # compute expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # optimize model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(game_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

### main training loop ###
num_games_train = 200

scores_train = []
lines_train = []
pieces_train = []
actions_train = []

for i_game in range(num_games_train):
    show = True#False
    #if i_game % 10 == 0:
    print("game", i_game)
    print(eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay))
        #show = True
    # initialize environment and state
    env = TetrisApp()
    state = torch.tensor(env.state, dtype=torch.float)
    score = 0
    for t in count():
        # select and perform action
        action = select_action(state)
        reward, done = env.step(action.item(), show)
        reward = torch.tensor([reward])
        score += gamma*reward

        # observe next state
        if not done:
            next_state = torch.tensor(env.state, dtype=torch.float)
        else:
            next_state = None
        # store transition in memory
        # including clean up of replay memory: fair representation of hard drops
        if action.item() == 5: # is hard drop
            memory.push(state, action, next_state, reward)
        else:
            p = random.random()
            if p < 0.02:
                memory.push(state, action, next_state, reward)

        state = next_state
        # perform one step of optimization (on target network)
        optimize_model()
        if done:
            game_rewards.append(score)
            plot_durations()
            break

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

plt.figure(1)
plt.clf()
plt.title('Actions - Training') # average number of actions per game
plt.xlabel('Game')
plt.ylabel('Moves')
plt.xticks(np.arange(len(actions_train)), np.arange(1, len(actions_train)+1))
plt.plot(actions_train)
#plt.axhline(avg_actions_train, color="#FFA500")
plt.locator_params(axis='x', nbins=10)
plt.show()

plt.figure(2)
plt.clf()
rewards_t = torch.tensor(game_rewards, dtype=torch.float)
plt.title('Rewards - Training')
plt.xlabel('Game')
plt.ylabel('Reward')
plt.plot(rewards_t.numpy())
# means = np.mean(rewards_t.numpy())
# plt.axhline(means, color="#FFA500")
plt.locator_params(axis='x', nbins=10)
plt.show()


### main testing loop ###
num_games_test = 100

scores_test = []
lines_test = []
pieces_test = []
actions_test = []
forces = []

for i_game in range(num_games_test):
    print("game", i_game)
    not_hard_drop = 0
    forced_action = 0
    # initialize environment and state
    env = TetrisApp()
    state = torch.tensor(env.state, dtype=torch.float)
    for t in count():
        # select and perform action
        #with torch.no_grad():
        q_vals = policy_net(state)
        action = q_vals.max(0)[1].view(1) # action with largest expected reward
        if action != 3:
            not_hard_drop += 1
        if not_hard_drop > 30: # force hard drop after 50 non hard drop actions
            action = torch.tensor([2], dtype=torch.long)
            not_hard_drop = 0
            forced_action += 1
        _, done = env.step(action.item(), False)

        # observe next state
        if not done:
            next_state = torch.tensor(env.state, dtype=torch.float)
        else:
            next_state = None
        state = next_state

        if done:
            break

    # performance metrics
    i_score_test, i_lines_test, i_pieces_test, i_actions_test = env.get_perf()
    scores_test.append(i_score_test)
    lines_test.append(i_lines_test)
    pieces_test.append(i_pieces_test)
    actions_test.append(i_actions_test)
    forces.append(forced_action)

print('Complete (testing)')

avg_scores_test = sum(scores_test)/len(scores_test)
avg_lines_test = sum(lines_test)/len(lines_test)
avg_pieces_test = sum(pieces_test)/len(pieces_test)
avg_actions_test = sum(actions_test)/len(actions_test)
perc_forces = sum(forces)/sum(pieces_test)

print("RESULTS - TESTING\n")
print("score:", avg_scores_test)
print("lines:", avg_lines_test)
print("pieces:", avg_pieces_test)
print("actions:", avg_actions_test)
print("forces:", perc_forces) # % of pieces forced to hard drop

plt.figure(3)
plt.clf()
plt.title('Actions - Testing') # average number of actions per game
plt.xlabel('Game')
plt.ylabel('Moves')
plt.xticks(np.arange(len(actions_test)), np.arange(1, len(actions_test)+1))
plt.plot(actions_test)
plt.axhline(avg_actions_test, color="#FFA500")
plt.locator_params(axis='x', nbins=10)
plt.show()

env.quit()
plt.ioff()
plt.show()
