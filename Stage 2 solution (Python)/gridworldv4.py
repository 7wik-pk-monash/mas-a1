# %%
import numpy as np
import random
import sys

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

def dir(start,end): #direction vector for two grid positions
    (x,y)=start
    (a,b)=end
    return (a-x,b-y)

class BoardPiece:

    def __init__(self, name, code, pos):
        self.name = name #name of the piece
        self.code = code #an ASCII character to display on the board
        self.pos = pos #2-tuple e.g. (1,4)
        self.loaded = False #flag whether carrying load or not

class BoardMask:

    def __init__(self, name, mask, code):
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self): #returns tuple of arrays
        return np.nonzero(self.mask)

def zip_positions2d(positions): #positions is tuple of two arrays
    x,y = positions
    return list(zip(x,y))

class GridBoard:

    def __init__(self, size=4):
        self.size = size #Board dimensions, e.g. 4 x 4
        self.components = {} #name : board piece
        self.masks = {}

    def addPiece(self, name, code, pos=(0,0)):
        newPiece = BoardPiece(name, code, pos)
        self.components[name] = newPiece

    #basically a set of boundary elements
    def addMask(self, name, mask, code):
        #mask is a 2D-numpy array with 1s where the boundary elements are
        newMask = BoardMask(name, mask, code)
        self.masks[name] = newMask

    def movePiece(self, name, pos):
        move = True
        for _, mask in self.masks.items():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
        if move:
            self.components[name].pos = pos

    def delPiece(self, name):
        del self.components['name']

    def render(self):
        dtype = '<U2'
        displ_board = np.zeros((self.size, self.size), dtype=dtype)
        displ_board[:] = ' '

        for name, piece in self.components.items():
            displ_board[piece.pos] = piece.code

        for name, mask in self.masks.items():
            displ_board[mask.get_positions()] = mask.code

        return displ_board

    def render_np(self):
        num_pieces = len(self.components) + len(self.masks)
        displ_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for name, piece in self.components.items():
            pos = (layer,) + piece.pos
            displ_board[pos] = 1
            layer += 1

        for name, mask in self.masks.items():
            x,y = self.masks['boundary'].get_positions()
            z = np.repeat(layer,len(x))
            a = (z,x,y)
            displ_board[a] = 1
            layer += 1
        return displ_board

def addTuple(a,b):
    return tuple([sum(x) for x in zip(a,b)])

# %%
class Gridworld:

    def __init__(self, size=4, mode='static'):
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GridBoard(size=4)

        #Add pieces, positions will be updated later
        self.board.addPiece('Player','P',(0,0))
        self.board.addPiece('Target','B',(1,0))
        self.board.addPiece('Source','A',(2,0))

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    #Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):
        #Setup static pieces
        self.board.components['Player'].pos = (0,3) #Row, Column
        self.board.components['Target'].pos = (0,0)
        self.board.components['Source'].pos = (self.board.size-1, self.board.size-1)

    #Check if board is initialized appropriately (no overlapping pieces)
    #also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = self.board.components['Player']
        target = self.board.components['Target']
        source = self.board.components['Source']

        all_positions = [piece for name,piece in self.board.components.items()]
        all_positions = [player.pos, target.pos, source.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0,0),(0,self.board.size), (self.board.size,0), (self.board.size,self.board.size)]
        #if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or target.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_move_go = [self.validateMove('Target', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                #print(self.display())
                #print("Invalid board. Re-initializing...")
                valid = False

        return valid

    #Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        #height x width x depth (number of pieces)
        self.initGridStatic()
        #place player
        self.board.components['Player'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    #Initialize grid so that all items are all randomly placed
    def initGridRand(self):
        #height x width x depth (number of pieces)
        self.board.components['Player'].pos = randPair(0,self.board.size)
        self.board.components['Source'].pos = randPair(0,self.board.size)
        self.board.components['Target'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def validateMove(self, piece, addpos=(0,0)):
        outcome = 0 #0 is valid, 1 invalid, 2 lost game
        #pit = self.board.components['Pit'].pos
        #wall = self.board.components['Wall'].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        #if new_pos == wall:
        #    outcome = 1 #block move, player can't move to wall
        if max(new_pos) > (self.board.size-1):    #if outside bounds of board
            outcome = 1
        elif min(new_pos) < 0: #if outside bounds
            outcome = 1
        #elif new_pos == pit:
        #    outcome = 2

        return outcome

    def makeMove(self, action):
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0,2]:
                new_pos = addTuple(self.board.components['Player'].pos, addpos)
                self.board.movePiece('Player', new_pos)

        if action == 'u': #up
            checkMove((-1,0))
        elif action == 'd': #down
            checkMove((1,0))
        elif action == 'l': #left
            checkMove((0,-1))
        elif action == 'r': #right
            checkMove((0,1))
        else:
            pass

    def reward(self):
      #BM
        if (not(self.board.components['Player'].loaded) and self.board.components['Player'].pos == self.board.components['Source'].pos):
            self.board.components['Player'].loaded=True
            return 50
        elif (self.board.components['Player'].loaded and self.board.components['Player'].pos == self.board.components['Target'].pos):
            self.board.components['Player'].loaded=False
            return 500
        else:
            return -1

    def display(self):
        return self.board.render()


    def state_np(self):
        player_state = self.board.components['Player'].loaded
        board_state = self.board.render_np()
        game_state = np.append(board_state.flatten(), player_state)
        return game_state

    def short_state_np(self):
        player = self.board.components['Player']
        loaded=  player.loaded
        target = self.board.components['Target']
        source = self.board.components['Source']
        # here we are using directions to A and B instead of their positions. Positions work in the same way but take longer to learn
        return np.array(list(player.pos)+list(dir(target.pos,player.pos))+list(dir(source.pos,player.pos))+[1 if loaded else 0],dtype=np.int8)

# %% [markdown]
# Try it out:

# %%
game = Gridworld(size=4, mode='static')

# %%
game.display()

# %%
game.board.render_np()

# %%
game.state_np()

# %%
game.short_state_np()

# %%
game.makeMove('d')
game.makeMove('d')
game.makeMove('l')
game.display()

# %%
game.reward()

# %%
game.state_np()

# %% [markdown]
# Test the reward when picking up the item

# %%
game.makeMove('r')
game.makeMove('d')
game.display()

# %%
game.reward()

# %%
game.board.components['Player'].loaded

# %%
game.state_np()

# %%
game.short_state_np()

# %% [markdown]
# Test the reward when dropping off the item

# %%
game.makeMove('u')
game.display()

# %%
game.reward()

# %%
game.makeMove('u')
game.makeMove('u')
game.makeMove('l')
game.makeMove('l')
game.display()

# %% [markdown]
# Still not quite there yet

# %%
game.reward()

# %%
game.state_np()

# %% [markdown]
# Now drop it

# %%
game.makeMove('l')
game.display()

# %%
game.reward()

# %%
game.board.render_np()

# %%
game.state_np()

# %%
game.short_state_np()

# %%
game.board.render_np().shape


