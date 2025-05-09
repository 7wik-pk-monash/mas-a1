import numpy as np

import random as rd

from stage2skeletonv3 import statespace_size

# north, east, south, west are the valid actions each agent can take
actions = ['n', 'w', 'e', 's']

def dir(start,end): #direction vector for two grid positions
    (x,y)=start
    (a,b)=end
    return ((a-x, b-y))

class Agent:

    def __init__(self, id, pos, reached_a, loc_a):

        self.id = id
        self.pos = pos

        # indicates if agent has already reached A in a run
        self.reached_a = reached_a

        if(loc_a == pos):
            self.reached_a = True

        self.num_collisions = 0
        self.num_steps = 0
        self.cycles = 0
    
    def _reset(self, pos, reached_a, loc_a):

        self.pos = pos
        self.reached_a = reached_a

        if(loc_a == pos):
            self.reached_a = True

        self.done = False
        self.num_collisions = 0
        self.num_steps = 0
        self.cycles = 0


class GridWorld:

    def __init__(self, n = 5, m = 5, loc_a = (0,0), loc_b = (4, 4), agents=None):
        
        # validations: n and m must be strictly positive, loc_a and b should be valid and not equal, number of agents shouldn't exceed the number of cells in the grid

        self.n = n
        self.m = m

        self.loc_a = loc_a
        self.loc_b = loc_b

        # each grid cell has NoneType objects by default
        self.grid = np.empty(shape=(n, m), dtype=object)

        # initialise every cell as an empty list
        for i in range(n):
            for j in range(m):
                self.grid[i, j] = []

        # add agents to some cells as list elements
        for agent in agents:
            self.grid[agent.pos[0], agent.pos[1]].append(agent)

    def _reset(self, loc_a, loc_b, agents):

        self.loc_a = loc_a
        self.loc_b = loc_b

        # each grid cell has NoneType objects by default
        self.grid = np.empty(shape=(self.n, self.m), dtype=object)

        # initialise every cell as an empty list
        for i in range(n):
            for j in range(m):
                self.grid[i, j] = []

        # add agents to some cells as list elements
        for agent in agents:
            self.grid[agent.pos[0], agent.pos[1]].append(agent)
    
    # returns a dict where key is agent id and element is the Agent object
    def get_agents_dict(self):

        agents_dict = {}

        for i in range(self.n):
            for j in range(self.m):
                if len(self.grid[i, j]) > 0:
                    for agent in self.grid[i, j]:
                        agents_dict[ agent.id ] = agent
        
        return agents_dict

    # should get an agent-specific game state in an np array representation to be used in training the DQN
    def get_np_state_for_agent(self, agent):

        # state space = agent_x X agent_y X loc_ax X loc_ay X loc_bx X loc_by X reached_a X neighbourhood_state (8 booleans)

        state = [agent.pos[0], agent.pos[1]] + list(dir(self.loc_a,agent.pos)) \
            + list(dir(self.loc_b,agent.pos)) + [int(agent.reached_a)]

        return np.array(state).reshape(1, statespace_size)


    # attempt to perform an action for a given agent if action is valid/permissible.
    # after performing this action, agent should be rewarded/penalised appropriately.
    # returns the reward/penalty
    def attempt_action_for_agent(self, agent, action):

        ## rewards/penalties

        boundary_pen = -24
        a_reach_rew = 24
        b_reach_rew = 24
        collision_pen = -24
        per_cycle_rew = 0
        step_pen = -0.5

        new_pos = agent.pos
        agent.num_steps += 1

        ## check if action is valid (maybe penalise for hitting a boundary)

        # boundary conditions

        match action:
            
            # agent.pos[0] - x coordinate - vertical conditions (north/south)
            # agent.pos[1] - y coordinate - horizontal conditions

            case 'n':

                if agent.pos[0] == 0:
                    return boundary_pen
                
                else:
                    new_pos = (agent.pos[0] - 1, agent.pos[1])

            case 'e':

                if agent.pos[1] == self.m - 1:
                    return boundary_pen
                
                else:
                    new_pos = (agent.pos[0], agent.pos[1] + 1)
            
            case 's':

                if agent.pos[0] == self.n - 1:
                    return boundary_pen
                
                else:
                    new_pos = (agent.pos[0] + 1, agent.pos[1])
            
            case 'w':

                if agent.pos[1] == 0:
                    return boundary_pen
                
                else:
                    new_pos = (agent.pos[0], agent.pos[1] - 1)

        ## perform the action

        self.grid[ agent.pos[0], agent.pos[1] ].remove(agent)
        self.grid[ new_pos[0], new_pos[1] ].append(agent)
        agent.pos = new_pos

        loc_reach_rew = step_pen

        ## check if agent reached A or B and reward
        if ( new_pos == self.loc_a ) and ( not agent.reached_a ):
            agent.reached_a = True
            agent.cycles += 1
            loc_reach_rew = a_reach_rew + (agent.cycles * per_cycle_rew)
        
        elif ( new_pos == self.loc_b ) and ( agent.reached_a ):
            # reached_a gets set to false as we need the agents to learn an infinite behavior - the agent will now head back to A
            agent.reached_a = False
            agent.cycles += 1
            loc_reach_rew = b_reach_rew + ( agent.cycles * per_cycle_rew )
        
        ## else: check head-on collisions and penalise
        else:
            agents_in_cell = self.grid[new_pos[0], new_pos[1]]
            for agent2 in agents_in_cell:
                if agent2.reached_a != agent.reached_a:
                    agent.num_collisions += 1
                    loc_reach_rew = collision_pen
                    break

        ## if it was a normal step, reward -1
        return loc_reach_rew

    def display(self):

        print(' -' + '-' * self.m * 3)

        for i in range(self.n):

            print('| ', end='')
            for j in range(self.m):

                if self.loc_a[0] == i and self.loc_a[1] == j:
                    print(' A ', end='')
                
                elif self.loc_b[0] == i and self.loc_b[1] == j:
                    print(' B ', end='')

                elif len(self.grid[i, j]) == 0:
                    print('   ', end='')

                if len(self.grid[i, j]) > 0:

                    if ((i, j) != self.loc_a) and ((i, j) != self.loc_b):

                        print(f'*{str(len(self.grid[i, j]))} ', end='')
                    
            
            print('|')
        
        print(' -' + '-' * self.m * 3)

        print(f"* followed by a number indicates the presence of a number of agents in that cell.\nagents at A and B are not indicated.\n")

        # debug print agents in the grid
        print("agents in the grid:")
        for id, ag in self.get_agents_dict().items():
            print(f"id: {id} pos: {ag.pos} is_loaded: {ag.reached_a}")
        print()

# initialise a number of agents randomly in either loc_a or loc_b
# loc_a and loc_b are coords of the form (x,y)
# returns a list of created agents of ids from 1 to num_agents
# and the first agent (id 1) is always placed at loc_b
def init_agents(num_agents, loc_a, loc_b):

    agents = []

    # for debugging only
    # debug_locs = [loc_a, loc_b, (loc_a[0] - 1, loc_a[1]), (loc_a[0] + 1, loc_a[1]), (loc_b[0] - 1, loc_b[1])]

    for i in range(num_agents):

        if i == 0:
            # at least one agent must begin at B
            agents.append( Agent(i+1, loc_b, False, loc_a) )
            continue
        
        # # debug force 2 at A
        # if i == 1:
        #     agents.append( Agent(i+1, loc_a, True, loc_a) )
        #     continue

        # rd_loc for each subsequent agent
        rd_loc = rd.choice([loc_a, loc_b])
        
        # debug
        # rd_loc = rd.choice(debug_locs)

        agents.append( Agent(i+1, rd_loc, (rd_loc==loc_a), loc_a) )
    
    return agents

def init_agents_random(n, m, num_agents, loc_a):

    agents = []

    for i in range(num_agents):
        pos = ( rd.randint(0, n-1), rd.randint(0, m-1) )
        agents.append( Agent( i+1, pos, rd.choice([True, False]), loc_a ))
    
    return agents

if __name__ == '__main__':

    n = 5
    m = 5
    num_agents = 4
    
    loc_a = (1,1)
    loc_b = (3,3)
    agents = init_agents(num_agents, loc_a, loc_b)

    gw = GridWorld(n, m, loc_a, loc_b, agents)
    gw.display()

    # debug print agents in the grid
    print("agents in the grid:")
    for id, ag in gw.get_agents_dict().items():
        print(id, ag.pos, ag.reached_a, end=' | ')
    print()

    # print("agent 1 np state:", gw.get_np_state_for_agent(agents[0]))
    # print()
    
    # print("agent 2 np state:", gw.get_np_state_for_agent(agents[1]))

    print(gw.attempt_action_for_agent(agents[0], 'w'))
    gw.display()

    print(gw.attempt_action_for_agent(agents[0], 's'))
    gw.display()


