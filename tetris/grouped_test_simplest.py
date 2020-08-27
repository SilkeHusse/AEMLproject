import torch
from itertools import count
import random
import torch.nn as nn
import torch.nn.functional as F
from tetris.grouped_tetris_simplest import *

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

policy_net = DQN()
policy_net.load_state_dict(torch.load("block_drop_eps_700_alt.pt"))

def select_action(states):
    global steps_done
    with torch.no_grad():
        vals = [policy_net(state) for state in states]
        return torch.tensor(np.argmax(np.array(vals))) # state with largest expected reward

### main testing ###
num_games_test = 1

scores_test = []
lines_test = []
pieces_test = []
actions_test = []

for i_game in range(num_games_test):
    # initialize environment and state
    env = TetrisApp()
    state = torch.tensor(env.state, dtype=torch.float)
    score = 0
    for t in count():
        # select and perform action
        states = [torch.tensor(s, dtype=torch.float) for s in env.get_next_states()]
        action = select_action(states)
        _, done = env.step_grouped(action.item(), True)

        # observe next state
        if not done:
            next_state = torch.tensor(env.state, dtype=torch.float)
        else:
            next_state = None

        state = next_state
        if done:
            break

    # performance metrics
    scores_test, lines_test, pieces_test, actions_test = env.get_perf()

print('Complete (testing)')

print("RESULTS - TESTING\n")
print("score:", scores_test)
print("lines:", lines_test)
print("pieces:", pieces_test)
print("actions:", actions_test)

env.quit()