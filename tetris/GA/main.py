
# tetris game and AI
from tetris_game import TetrisApp
from tetris_game_half import TetrisApp as TetrisAppHalf
from tetris_ai import TetrisAI
import threading

"""
### Training the genetic algorithms

# Half Tetris
app = TetrisAppHalf()
ai = TetrisAI(app)

threading.Thread(target=app.run).start()
ai.start(num_units=200, max_gen=20,elitism_rate=0.4, crossover_rate=0.4, mutation_val=0.1, target_file="tetrishalf.csv")


# Tetris Limited stones
app = TetrisApp()
ai = TetrisAI(app)

threading.Thread(target=app.run).start()
ai.start(max_stones=1000, num_units=200, max_gen=20,elitism_rate=0.4, crossover_rate=0.4, mutation_val=0.1, target_file="tetrislim.csv")
"""



### Testing on the full field without restrictions
app = TetrisApp()
ai = TetrisAI(app)
threading.Thread(target=app.run).start()

# Tetris Half candidates
ai.start(seed=(-0.9533557179974987, -0.32412555725603726, -0.8068493661861214, 0.7173954980057654))
ai.start(seed=(-0.8559213720198857, -0.26642993295988393, -0.24165679838256565, 0.7604564183939784))
ai.start(seed=(-0.8520418915733747, -0.2766222767928419, -0.7746061969834042, 0.8147079138574993))

# Tetris Limited candidates
ai.start(seed=(-1.0535666763369844, -0.026840703258964815, 0.014439157431660687, 0.02982986409333721))
ai.start(seed=(-0.9820328878548452, -0.2522080657773923, -0.6964781505718165, 0.3110665417218754))
ai.start(seed=(-0.8528557987064251, -0.26968611632170625, -0.9119192556266463, 0.4806613199270128))


