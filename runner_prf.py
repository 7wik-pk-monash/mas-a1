from grid_world import *
from stage2skeletonv3 import *
import random as rd

from collections import deque
from IPython.display import clear_output

def TD_target(reward, gamma, done, qMax):
    # done was an arg in skeleton
#   target = reward + gamma*(1-done)*qMax
    target = reward + gamma*(1-done)*qMax
    return target


losses=[]
collisions_lst = []
a_b_switches_lst = []
action_set = {0: 'n', 1: 'w', 2: 'e', 3: 's'}
j= 0
collisions = 0
total_a_b_switches = 0

def train():
    # 4 neighbours - statespace 11, 8 neighbours - 15
    statespace_size = 11 # 3 (x,y) coords + a boolean (reached_a) + 8 neighbourhood booleans (presence of neighbours of opposite type - different reached_a)

    global losses, collisions_lst, a_b_switches_lst, action_set, model, j, collisions, total_a_b_switches

    gamma = 0.95
    # epsilon = 1.0
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995

    epochs = 1000
    mem_size = 5000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 100
    h = 0
    sync_freq = 500
    j=0
    rewards=[]

    model = prepare_torch()

    n = 5
    m = 5

    num_agents = 4

    total_a_b_switches = 0
    collisions = 0

    # initialise all configs and agents to optimize training performance

    loc_as = []
    loc_bs = []
    agents_each_epoch = []
    gws = []

    for i in range(epochs):
        
        loc_a = (rd.randint(0,n-1), rd.randint(0,m-1))
        loc_as.append(loc_a)
        loc_b = (rd.randint(0,n-1), rd.randint(0,m-1))

        # ensuring loc_b and loc_a aren't equal
        while loc_b == loc_a:
            loc_b = rd.randint(0,n-1), rd.randint(0,m-1)
        
        loc_bs.append(loc_b)
        agents = init_agents( num_agents, loc_a, loc_b )
        agents_each_epoch.append( agents )
        # agents_each_epoch.append( init_agents_random( n, m, num_agents, loc_a ) )

        gws.append( GridWorld(n, m, loc_a, loc_b, agents) )

    # main loop 

    for i in range(epochs):

        loc_a = loc_as[i]
        loc_b = loc_bs[i]

        agents = agents_each_epoch[i]

        # a_b_switches = {}
        # for agent in agents:
        #     a_b_switches[agent.id] = 0

        gw = gws[i]

        steps = 0
        loop = True
        episode_reward = 0
        epoch_a_b_switches = 0
        epoch_collisions = 0

        while loop and (steps < max_moves):

            # # change loop_agents to random order
            loop_agents = agents

            # rd.shuffle(loop_agents)
            
            for agent in loop_agents:

                # print(steps)
                init_reached_a = agent.reached_a

                state1 = gw.get_np_state_for_agent(agent)
                steps += 1
                j += 1
                qvals = get_qvals(state1)

                if (rd.random() < epsilon):
                    action_ = rd.randint(0,3)
                else:
                    action_ = np.argmax(qvals)
                
                action = action_set[action_]
                
                reward = gw.attempt_action_for_agent(agent, action)

                state2 = gw.get_np_state_for_agent(agent)

                episode_reward += reward

                if agent.reached_a != init_reached_a:
                    epoch_a_b_switches += 1

                # if we consider pickup and dropoff as terminal condition - we dont - because agents need to learn infinite behavior
                # done = True if (a_b_switches[agent.id] >= 2) else False # both pickup and delivery at least once
                
                # if we consider 1 collision 
                # or num_steps over 25
                # as a terminal condition
                done = False
                if (agent.num_collisions > 0) or (agent.num_steps > 25):
                    done = True
                    loop = False

                exp = (state1, action_, reward, state2, done)
                replay.append(exp)

                if (len(replay) > batch_size) and (j % 10 == 0):

                    minibatch = rd.sample(replay, batch_size)
                    states = [s1 for (s1,_,_,_,_) in minibatch]
                    actions = [a for (_,a,_,_,_) in minibatch]
                    targets = [TD_target(r, gamma, done, get_maxQ(s2)) for (_,_,r,s2,_) in minibatch]
                    current_loss = train_one_step(states, actions, targets, gamma)
                    losses.append(current_loss)
                    # print(i, "a_b_switches:", total_a_b_switches, " loss:", current_loss, "total collisions:", collisions, "epoch_collisions:", epoch_collisions)
                    # print(f"episode_reward: {episode_reward} steps: {steps} epoch_a_b_switches: {epoch_a_b_switches} epsilon: {epsilon}")
                    # print()
                    # clear_output(wait=True)

                if j % sync_freq == 0:
                    update_target()
            
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            epoch_collisions = sum([agent.num_collisions for agent in loop_agents])

            # final - pending work
            # if (steps > max_moves):
            # debug
            # if (episode_reward >= 150) or (episode_reward <= -2000):
            # if ( epoch_successes >= 8 ) or (steps > max_moves) \
            # or ( epoch_collisions >= 1 ) :
            #     rewards.append(episode_reward)  
            #     loop = False
            #     steps = 0
            #     break

            if steps > max_moves:
                rewards.append(episode_reward)
                loop = False
                steps = 0
                break
            
            # print(i)
            # clear_output(wait=True)

        total_a_b_switches += epoch_a_b_switches

        for agent in agents:
            collisions += epoch_collisions
        
        collisions_lst.append(epoch_collisions)
        a_b_switches_lst.append(epoch_a_b_switches)
        
        # clear_output(wait=True)  

    # print(f"total steps in training: {j}, total collisions: {collisions}, total a/b switches: {total_a_b_switches}")

# train()

import cProfile
cProfile.run("train()")