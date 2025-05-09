import numpy as np

action_set = {0: 'n', 1: 'w', 2: 'e', 3: 's'}

def init_q_table(n,m,):

    # 11 dims
    # 3 (x,y) coords + a boolean (reached_a) + 4 or 8 neighbourhood booleans (presence of neighbours of opposite type - different reached_a)
    # q_table = np.zeros(shape=(n,m,n,m,n,m,2,2,2,2,2))
    
    # no neighbors
    return np.zeros(shape=(n,m,n,m,n,m,2,4))

def update_q(q_table, state, action, next_state, reward, learning_rate, discount_factor, done):

    max_q_value = 0 if done else np.max(q_table[next_state])

    q_table[state + (action,)] += learning_rate * (reward + discount_factor * max_q_value - q_table[state + (action,)])

def choose_action(q_table, state, epsilon):

    if np.random.uniform(0,1) < epsilon:
        action = np.random.randint(4)
    else:
        # deb = q_table[state]
        action = np.argmax(q_table[state])
    
    return action