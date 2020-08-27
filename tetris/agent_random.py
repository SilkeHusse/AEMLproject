from tetris.tetris_random import *
import matplotlib.pyplot as plt
import numpy as np

# testing loop
num_games = 100
scores = []
lines = []
pieces = []
actions = []

for i_games in range(num_games):
    print("game", i_games)
    env = TetrisApp() # initialize environment
    env.run(False) # play game
    i_score, i_lines, i_pieces, i_actions = env.get_perf()
    scores.append(i_score)
    lines.append(i_lines)
    pieces.append(i_pieces)
    actions.append(i_actions)

avg_score = sum(scores)/len(scores)
avg_lines = sum(lines)/len(lines)
avg_pieces = sum(pieces)/len(pieces)
avg_actions = sum(actions)/len(actions)

plt.figure(1)
plt.clf()
plt.title('Actions') # average number of actions per game
plt.xlabel('Game')
plt.ylabel('Moves')
plt.xticks(np.arange(len(actions)), np.arange(1, len(actions)+1))
plt.plot(actions)
plt.axhline(avg_actions, color="#FFA500")
plt.locator_params(axis='x', nbins=10)
plt.show()

print("RESULTS\n")
print("score:", avg_score)
print("lines:", avg_lines)
print("pieces:", avg_pieces)
print("actions:", avg_actions)

