from grid_world import *
from stage2skeletonv3 import *
import random as rd

statespace_size = 15 # 3 (x,y) coords + a boolean (reached_a) + 8 neighbourhood booleans (presence of neighbours of opposite type - different reached_a)

gamma = 0.95
epsilon = 1.0
epsilon_min=0.01
epsilon_decay=0.995

def TD_target(reward, gamma, done, qMax):
  target = reward + gamma*(1-done)*qMax
  return target

action_set = {0: 'n', 1: 'e', 2: 's', 3: 'w'}

from collections import deque
from IPython.display import clear_output

epochs = 10000
mem_size = 2000
batch_size = 128
replay = deque(maxlen=mem_size)
max_moves = 100
h = 0
sync_freq = 250
j=0
losses=[]
rewards=[]
successes=0

model = prepare_torch()


n = 5
m = 5

num_agents = 4
    
loc_a = (3,1)
loc_b = (3,2)
agents = init_agents(num_agents, loc_a, loc_b)

# debug print agents in the grid
print("agents in the grid:")
for ag in agents:
    print(ag.id, ag.pos, ag.reached_a, end=' | ')
print()

# # debug print agent neighbourhoods
# a1n = gw.get_agents_neighbourhood(agent=agents[0])
# a2n = gw.get_agents_neighbourhood(agent=agents[1])

gw = GridWorld(n, m, loc_a, loc_b, agents)
gw.display()

# debug print agents in the grid
print("agents in the grid:")
for id, ag in gw.get_agents_dict().items():
    print(id, ag.pos, ag.reached_a, end=' | ')
print()

# main loop

steps = 0
loop = True
episode_reward = 0
successes = 0

states1 = dict(zip([agent.id for agent in agents], [gw.get_np_state_for_agent(agent).reshape(1, statespace_size) for agent in agents]))

while loop:

    # # change loop_agents to random order
    loop_agents = agents
    max_reward = 0
    
    for agent in loop_agents:

        state1 = states1[agent.id]
        steps += 1
        j += 1
        qvals = get_qvals(state1)

        if (rd.random() < epsilon):
            action_ = rd.randint(0,3)
        else:
            action_ = np.argmax(qvals)
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        action = action_set[action_]
        
        reward = gw.attempt_action_for_agent(agent, action)

        if reward > max_reward:
            max_reward = reward

        state2 = gw.get_np_state_for_agent(agent).reshape(1, statespace_size)

        episode_reward += reward

        done = True if reward > 50 else False # pickup and delivery once
        if done:
            successes += 1

        exp = (state1, action_, reward, state2, done)
        replay.append(exp)

        states1[agent.id] = state2
    
    if len(replay) > batch_size:
        minibatch = np.array(rd.sample(replay, batch_size))
        states = minibatch[...,0]
        actions = minibatch[...,1]
        targets = [TD_target(r, gamma, d, get_maxQ(s2)) for (_,_,r,s2,d) in minibatch]
        current_loss = train_one_step(states, actions, targets, gamma)
        losses.append(current_loss)
        clear_output(wait=True)
        print(1, " successes:", successes, " loss=", current_loss)
        print(f"episode reward: {episode_reward} mov: {steps}")
        print()
        if j % sync_freq == 0:
            update_target()

    if reward == 500 or steps > max_moves:
        rewards.append(episode_reward)  
        loop = False
        steps = 0
    
    # debug
    if steps == 20:
        loop = False
        break

    

from matplotlib import pylab as plt

plt.xlabel("Episode",fontsize=22)
plt.ylabel("Reward",fontsize=22)
plt.plot(np.array(losses))
plt.figure(figsize=(10,7))


### Runner skeleton
# 
# Runner should probably have agent-specific choose_action logic 
# 
# for each epoch, start with a gw
# for this gw, per each agent (in random order)...
#   get agent-specific "state" as state1 from gw (should have encoded neigbourhood values)
#   get qvals
#   get max qval and take the corresponding action for this agent
#   use reward returned by attempt_action...() to construct new state (state2) and store it back in the deque
#   state1 = state2, repeat for next random agent
# 
#

# class Runner:

#     def __init__(self, gw, agents):

#         self.gw = gw
#         self.agents = agents

#     def run(self):
#        pass
