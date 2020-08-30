###
# The structure of the basic python implementation for this AI is inspired by 
# Thomas Young (https://github.com/thyo9470/Genetic-Tetris).
# The core functions were changed to our needs, new options and functions were added.


from collections import defaultdict, OrderedDict
import pygame, numpy, random, sys, math, csv, re

# Rotates a shape clockwise
def rotate_clockwise(shape):
  return [ [ shape[y][x]
      for y in range(len(shape)) ]
    for x in range(len(shape[0]) - 1, -1, -1) ]

# checks if there is a collision in any direction
def check_collision(board, shape, offset):
  off_x, off_y = offset
  for cy, row in enumerate(shape):
    for cx, cell in enumerate(row):
      try:
        if cell and board[ cy + off_y ][ cx + off_x ]:
          return True
      except IndexError:
        return True
  return False

# Used for adding a stone to the board
def join_matrixes(mat1, mat2, mat2_off):
  off_x, off_y = mat2_off
  for cy, row in enumerate(mat2):
    for cx, val in enumerate(row):
      try:
        mat1[cy+off_y-1  ][cx+off_x] += val
      except IndexError:
        print("out of bounds join")
  return mat1


    

### Tetris AI
class TetrisAI(object):
  
  def __init__(self, tetris_app):
    self.name = "Crehg"
    self.tetris_app = tetris_app

    self.screen = pygame.display.set_mode((200, 480 ))

    # set fetures wanted here 
    self.features = ("cumulative_height", "roughness", "hole_count", "rows_cleared")

  def draw_matrix(self, matrix, offset, color=(255,255,255)):
    off_x, off_y  = offset
    for y, row in enumerate(matrix):
      for x, val in enumerate(row):
        if val:
          pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(
              (off_x+x) *
                20,
              (off_y+y) *
                20, 
              20,
              20),0)

# Getters and Setters
  def set_weights(self, weight_dict):
    self.weights = defaultdict(int, weight_dict)

  def set_board(self, board):
    self.board = board

  def get_board(self):
    if not hasattr(self, "board"):
      raise ValueError("TerisAI does not have a board")

    return self.board

  def set_stone(self, stone, stone_x, stone_y):
    self.stone = stone
    self.stone_x = stone_x
    self.stone_y = stone_y

  def get_stone(self):
    if not hasattr(self, "stone"):
      raise ValueError("TertisAI does not have a stone")

    return (self.stone, self.stone_x, self.stone_y)

  

  def make_move(self, num_stones=0, training=True):
    """
      Action Control, decides which moves to make, and how to go on after the game is finished
      (either by gameover of after a maximal number of stones is reached)

      Parameters
      ----------
      num_stones : int, optional
          stone count to account for a maximal stone number per game. The default is 0.
      training : boolean, optional
          to decide wether we want to train, or to test a specific seed. The default is True.
    """
    while True:
      
      cur_state = self.tetris_app.get_state()
      
      self.set_board(cur_state["board"])
      self.set_stone(cur_state["stone"], cur_state["stone_x"], cur_state["stone_y"])
      
      
      if not cur_state["needs_actions"]:
        continue

      actions = []
      
      # if gameover is reached, start with next unit
      if cur_state["gameover"] and training:
        self.load_next_unit( cur_state["score"], cur_state["lines"] )
        num_stones = 0
        actions.append("space")
       
      # for testing specific seeds: play the game 20 times with the same seed  
      if cur_state["gameover"] and training==False:
          self.info.append([cur_state["lines"], cur_state["score"]])
          num_stones = 0
          actions.append("space")
          self.tetris_app.add_actions(actions)
          while len(self.info)<20:
              self.load_weights(self.seed)
              self.make_move(training=False)
          break  
      
      
      # check all possible moves and chose the best
      possible_boards = self.get_possible_boards()
      board_scores = self.get_board_scores(possible_boards)
      actions.extend(self.get_actions_from_scores(board_scores))
     
      self.tetris_app.add_actions(actions)
      
      # count the stones
      if num_stones <= self.max_stones:
        num_stones +=1
      
      # termination condition: if we reach max_stones
      if num_stones > self.max_stones:
        print("stone limit reached")
        self.tetris_app.set_gameover()
      



  # move a piece horizontally
  def move(self, desired_x, board, stone, stone_x, stone_y):

    while(stone_x != desired_x):
      dist = desired_x - stone_x
      delta_x = int(dist/abs(dist))

      new_x = stone_x + delta_x
      if not check_collision(board,
                             stone,
                             (new_x, stone_y)):
        stone_x = new_x
      else:
        break
    return stone_x
 
# rotate a stone if no collision
  def rotate_stone(self, board, stone, stone_x, stone_y):

    new_stone = rotate_clockwise(stone)
    if not check_collision(board,
                           new_stone,
                           (stone_x, stone_y)):
      return new_stone
    return stone

 
  # drop piece downwards as long as no collision occurs. 
  # If collision: add stone to board, check for row completion.
  def drop(self, board, stone, stone_x, stone_y):

    stone_y += 1
    if check_collision(board,
                       stone,
                       (stone_x, stone_y)):
      board = join_matrixes(
        board,
        stone,
        (stone_x, stone_y))
    else:
      self.drop(board, stone, stone_x, stone_y)
    return board, stone_y

# get all possible boards
  def get_possible_boards(self):
    if not (hasattr(self, "board") and hasattr(self, "stone")):
      raise ValueError("either board or stone do not exist for TetrisAI")
    
    cur_state = self.tetris_app.get_state()

    self.set_board(cur_state["board"])
    self.set_stone(cur_state["stone"], cur_state["stone_x"], cur_state["stone_y"])

    temp_board = numpy.copy(self.board)
    temp_stone = numpy.copy(self.stone)

    temp_x = self.stone_x
    temp_y = self.stone_y

    # contains all the board orientations possible with the current stone
    boards = []

    for j in range(4):


      for i in range(len(self.board[0])):

        temp_x = self.move(i, temp_board, temp_stone, temp_x, temp_y)
        temp_board, temp_y = self.drop(temp_board, temp_stone, temp_x, temp_y)


        boards.append(temp_board)

        temp_board = numpy.copy(self.board)
        temp_x = self.stone_x
        temp_y = self.stone_y

      temp_stone = self.rotate_stone(temp_board, temp_stone, temp_x, temp_y)

    return boards

  # get the scores of all possible boards
  def get_board_scores(self, boards):
    scores = []

    for board in boards:
      new_score = self.eval_board(board)
      scores.append( new_score )

    return scores 

  # decide which option to choose from all possible boards
  def get_actions_from_scores(self, scores):
    actions = []

    # best score
    best_score = scores.index( max(scores) )

    # rotate to proper orientation
    rotations = [ "up" for i in range( best_score//len(self.board[0]) )]
    actions.extend(rotations)

    # move to proper x pos
    desired_x = best_score % len(self.board[0]) 
    wanted_move = ""  
    if( desired_x > self.stone_x ):
      wanted_move = "right" 
    elif(desired_x < self.stone_x):
      wanted_move = "left"
    
    moves = [ wanted_move for i in range( abs(desired_x - self.stone_x) ) ]
    actions.extend(moves)

    # move to proper y pos
    levels =  self.get_column_heights(self.board)
    desired_y = levels[desired_x] + len(self.stone) 

    world_y = len(self.board) - self.stone_y
    num_drops = world_y - desired_y
    drops = [ "down" for i in range( num_drops ) ]
    #actions.extend(drops)

    return actions

  # evaluate boards, fitness function (linear)
  def eval_board(self, board):

    if not (hasattr(self, "weights")):
      raise ValueError("TetrisAI has no weights")


    # all features in a linear function
    score = []
    score.append( self.get_cumulative_height(board) * self.weights["cumulative_height"])
    score.append( self.get_roughness(board) * self.weights["roughness"])
    score.append( self.get_hole_count(board) * self.weights["hole_count"])
    score.append( self.get_rows_cleared(board) * self.weights["rows_cleared"])

    return sum(score)

  #get height of a column
  def get_column_heights(self, board):
    # get the hights of each column
    heights = [0 for i in board[0]] 

    for y, row in enumerate(board[::-1]):
      for x, val in enumerate(row):
        if val != 0:
          heights[x] = y
    
    return heights


  # get sum of all column heights
  def get_cumulative_height(self, board):
    return sum(self.get_column_heights(board))



  # get roughness of the board (sum of the differences in height of neighboring columns)
  def get_roughness(self, board):

    levels = self.get_column_heights(board)

    # get roughness
    roughness = 0

    for x in range(len(levels)-1):
      roughness += abs(levels[x] - levels[x+1]) 

    return roughness

  # get the number of holes (tile above has to be blocked)
  def get_hole_count(self, board):
    levels = self.get_column_heights(board) 

    holes = 0

    for y, row in enumerate(board[::-1]):
      for x, val in enumerate(row):
        # if below max column height and is a zero
        if y < levels[x] and val==0:
          holes += 1 

    return holes

  # count the number of rows cleared
  def get_rows_cleared(self, board):
    # starts at -1 to account for bottom row which
    # is always all 1
    rows_cleared = -1
    
    for row in board:
      if 0 not in row:
        rows_cleared += 1
     
    return rows_cleared 


### GENETIC ALGORITHM


  #Creates a gene with random weights (between -1 and 1)
  #If seeded: It creates the weights based off the seeded gene with a certain variation 
  def random_weights(self, seeded=False):
   
    weights = ()

    if seeded != False:
      for val in seeded:
        weights = weights + (random.uniform(-0.1, 0.1) + val,)
      return weights

    for f in self.features:
      weights = weights + (random.uniform(-1, 1),)
 
    return weights
  
  # load weights
  def load_weights(self, weight_tuple):
    self.weights = dict()

    for fn, f in enumerate(self.features):
      self.weights[f] = weight_tuple[fn]
      

  def start(self, num_units=50, max_gen=30, max_stones=math.inf,elitism_rate=0.3, crossover_rate=0.4, mutation_val=0.05, target_file= "data.csv", seed=False):
    '''
      initialize the Genetic Algorithm. The starting population is 10 times the size of num_units 
      (=every further generation). A new generation is build by elitism selection (best candidates), 
      crossover (from randomly chosen candidates, weighted by their score) and mutation 
      (mutated by the mutation_val, chosen from the so far selected candidates from the new generation).

      Parameters
      ----------
      num_units : int, optional
          size of the popultion. The default is 50.
      max_gen : int, optional
          maximal number of generations. The default is 30.
      max_stones : int, optional
          maximal number of tetriminos that are available in each game. The default is math.inf.
      elitism_rate : float, optional
          rate of eltisim selection. The default is 0.3.
      crossover_rate : float, optional
          rate of crossover. The default is 0.4.
      mutation_val : float, optional
          how strong the mutation candidates are mutated. The default is 0.05.
      target_file : string, optional
          file direction where to save all information (generation, unit, weights, lines cleared,
          scores). The default is "data.csv".
      seed : tuple, optional
          To test a specific weight set. The default is False.
    '''
    # if seed: test this seed in 20 games
    if seed:
      if not (isinstance(seed, tuple) or len(seed) != len(self.features)):
        raise ValueError('Seed not properly formatted. Make sure it is a tuple and has {} elements').format(len(self.features))
      self.seed=seed
      self.max_stones = max_stones
      self.info =[]
      self.load_weights(seed)
      self.make_move(training=False)
      print(self.info)
      avg_lines =sum(x[0] for x in self.info)/len(self.info)
      avg_score = sum(x[1] for x in self.info)/len(self.info)
      print("avg. number of lines cleared:", avg_lines)
      print("avg. score:", avg_score)
    # train
    else:
      self.target_file = target_file
      if not bool(re.search(r".+(\.csv)$", target_file)):
          raise ValueError("target file has to be a csv file and has to be given as a string")
      self.num_units = num_units
      self.max_gen = max_gen
      self.max_stones = max_stones
      self.elitism_rate = elitism_rate
      self.crossover_rate = crossover_rate
      if not elitism_rate*10 in range(0 ,10) or not crossover_rate*10 in range(0 ,10):
          raise ValueError("rates have to be a float bewteen [0,1] (stepsize 0.1)")
      self.gen_scores = []
      self.info = []
      self.gen_weights = OrderedDict()
      self.cur_gen = 1
      self.cur_unit = -1
      self.mutation_val = mutation_val
      
      # starting populition with a 10 times greater size
      for i in range(num_units * 10):
        self.gen_weights[ self.random_weights() ] = 0
      
      # start the training
      self.load_next_unit(0, 0)
      self.make_move()


  #Saves data from previous geneartion, preforms eltisim selection, crossover, and mutation
  def new_generation(self, weight_values):

    
    print("New Generation") 
    print("\n\n")

    gen_keys = list(self.gen_weights.keys())
    
    # create a weights and a candidates list (pre-ordered)
    candidates = [ gen_keys[tup[0]] for tup in weight_values]
    weights = []
    for i in range(len(weight_values)):
        weights.append(weight_values[i][1])
        
    
    # elitism selection: select the top x% candidates and (a) take them directly to the new generation
    new_gen = []
    selected_units = candidates[:int(round(self.num_units*self.elitism_rate))]
    for i in range( len(selected_units)-1 ):
        new_gen.append(selected_units[i])
    
    # add the selected genes to the new gen_weights (new generation)
    self.gen_weights = OrderedDict()
    for new_unit in new_gen:
        self.gen_weights[ new_unit ] = 0
    
    # crossover: fill up the next part of the new generation with candidates - passed on
    # via crossover from the weighted candidates (while preventing duplicated candidates)
    while len(self.gen_weights) < self.num_units*(self.elitism_rate+self.crossover_rate):
      
      unit1, unit2 = random.choices(candidates,weights , k=2)
      new_unit1, new_unit2 = self.mix_genes(unit1, unit2)
      
      # add the new genes to gen_weights
      self.gen_weights[new_unit1] = 0
      self.gen_weights[new_unit2] = 0
    
    # mutation: fill up the rest of the new gen with mutated genes (chosen from the selected candidates so far)
    gen_so_far = list(self.gen_weights.keys())
    while len(self.gen_weights) < self.num_units:
        self.gen_weights[self.mutate_gene(random.choice(gen_so_far))] = 0
    
    # set the unit count back to zero
    self.cur_unit = 0

  # crossover function
  def mix_genes(self, gene1, gene2):
    if(len(gene1) != len(gene2)):
      raise ValueError('A very specific bad thing happened.') 

    num_features = len(self.features)
    new_genes_to_switch = numpy.random.choice( range(num_features), num_features//2, replace=False )  

    new_gene1 = ()
    new_gene2 = ()
    
    # mix the genes
    for i in range( len(gene1) ):
      if i in new_genes_to_switch:
        new_gene1 = new_gene1 + ( gene2[i], )
        new_gene2 = new_gene2 + ( gene1[i], )
      else:
        new_gene1 = new_gene1 + ( gene1[i], )
        new_gene2 = new_gene2 + ( gene2[i], )
       
    return (new_gene1, new_gene2) 

  # mutation function
  def mutate_gene(self, gene):

    num_features = len(self.features)
    genes_to_mutate = numpy.random.choice( range(num_features), random.randint(0, num_features), replace=False )
    new_gene = ()
    
    # mutate
    for i in range(len(gene)):
      mut_val = 0 
      if i in genes_to_mutate:
        mut_val = random.uniform(-self.mutation_val, self.mutation_val)
      new_gene = new_gene + ( gene[i] + mut_val,)
      
    return new_gene

  # load a gene into the ai to be used for Tetris
  def load_next_unit(self, score, lines):
    
    # evaluate current unit
    if self.cur_unit in range(0,len(self.gen_weights)):
      cur_weight = list(self.gen_weights.keys())[self.cur_unit]
      self.gen_weights[cur_weight] = score
      print("Gen: ", self.cur_gen,"|| Unit: ", self.cur_unit)
      print("weights:", cur_weight)
      print("lines cleared: ", lines)
      print("score: ", score)
      print("--------------------------------------------------")
      self.info.append([self.cur_gen, self.cur_unit, cur_weight, lines, score])
      

    self.cur_unit += 1
    
    # create new unit in current generation
    if self.cur_unit < len(self.gen_weights):
        new_unit = list(self.gen_weights.keys())[self.cur_unit]
        self.load_weights(new_unit)
        
    # evaluate current generation and create new generation if the current generation is finished
    elif self.cur_unit == len(self.gen_weights):
        weight_values = sorted( enumerate(self.gen_weights.values()), key= lambda x:x[1], reverse=True)
        print("\n\n")
        gen_score = sum(x[1] for x in weight_values)/len(weight_values)
        self.gen_scores.append(gen_score)
        print("Generation Scores:", self.gen_scores)
        
        # termination condition: no progress between the last four generations
        if len(self.gen_scores) >= 4:
            for i in range(len(self.gen_scores)-3):
                if self.gen_scores[i] >  self.gen_scores[i+1] and self.gen_scores[i+1] >  self.gen_scores[i+2] and self.gen_scores[i+2] >  self.gen_scores[i+3]:
                    print("No more progress")
                    # csv writer
                    f = open(self.target_file, "w")
                    with f:
                        writer = csv.writer(f)
                        for row in self.info:
                            writer.writerow(row) 
                    sys.exit()
                    
        # create new generation if max generation is not reached yet           
        if self.cur_gen < self.max_gen:
            self.cur_gen += 1
            self.new_generation(weight_values)
            
        # termination condition: reached max generation
        else:
            print("Done playing tetris")
            # csv writer
            f = open(self.target_file, "w")
            with f:
                writer = csv.writer(f)
                for row in self.info:
                    writer.writerow(row) 
            sys.exit()
    
    else:
        print("Something went wrong, unit count to high")
        sys.exit()
        
      

   
    

