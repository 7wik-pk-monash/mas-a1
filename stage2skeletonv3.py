# %%
import torch
import copy
import numpy as np

# %% [markdown]
# You need to instantiate the constant for the size of the state space below. This will be used as the size of the input tensor for your Q-network

# %%
# statespace_size = 11
statespace_size = 7
# learning_rate = 1e-3
learning_rate = 2e-4

# debug
global_loc_a = (1,0)
global_loc_b = (4,4)

# %% [markdown]
# The function "prepare_torch" needs to be called once and only once at the start of your program to initialise PyTorch and generate the two Q-networks. It returns the target model (for testing).

# %%
def prepare_torch():
  global statespace_size
  global model, model2
  global optimizer
  global loss_fn
  global learning_rate
  l1 = statespace_size
  l2 = 24
  l3 = 24
  l4 = 4
  model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4))
  model2 = copy.deepcopy(model)
  model2.load_state_dict(model.state_dict())
  loss_fn = torch.nn.MSELoss()
  # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  return model2

# def TD_target(reward, gamma, done, qMax):
#     # done was an arg in skeleton
# #   target = reward + gamma*(1-done)*qMax
#     target = reward + gamma*(1-done)*qMax
#     return target

def TD_target(rewards, gamma, dones, q_max_next_states):
    """
    Calculates the TD target in a vectorized manner.

    Args:
        rewards (np.ndarray or torch.Tensor): Rewards for each transition.
        gamma (float): Discount factor.
        dones (np.ndarray or torch.Tensor): Boolean flags indicating if the episode is done.
        q_max_next_states (np.ndarray or torch.Tensor): Maximum Q-values for the next states.

    Returns:
        torch.Tensor: The calculated TD targets.
    """

    # Ensure inputs are tensors for consistency and operations
    if not isinstance(rewards, torch.Tensor):
        # print(type(rewards))
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if not isinstance(dones, torch.Tensor):
        dones = torch.tensor(dones, dtype=torch.float32)
    if not isinstance(q_max_next_states, torch.Tensor):
        q_max_next_states = torch.tensor(q_max_next_states, dtype=torch.float32)

    targets = rewards + gamma * (1 - dones) * q_max_next_states
    return targets

# %% [markdown]
# The function "update_target" copies the state of the prediction network to the target network. You need to use this in regular intervals.

# %%
def update_target():
  global model, model2
  model2.load_state_dict(model.state_dict())

# %% [markdown]
# The function "get_qvals" returns a numpy list of qvals for the state given by the argument _based on the prediction network_.

# %%
def get_qvals(state):
  state1 = torch.from_numpy(state).float()
  qvals_torch = model(state1)
  qvals = qvals_torch.data.numpy()
  return qvals

# %% [markdown]
# The function "get_maxQ" returns the maximum q-value for the state given by the argument _based on the target network_.

def get_maxQ(next_states):
    """
    Calculates the maximum Q-values for a batch of next states.

    Args:
        next_states (np.ndarray): A batch of next states.

    Returns:
        np.ndarray: An array of maximum Q-values for each next state.
    """

    next_states_tensor = torch.from_numpy(next_states).float()  # Convert to PyTorch tensor
    with torch.no_grad():  # Important: Disable gradient calculation
        q_values = model2(next_states_tensor)  # Shape: (batch_size, 1, action_space_size)
        max_q_values, _ = torch.max(q_values.squeeze(1), dim=1)  # Squeeze the middle dim, then get max
    return max_q_values.numpy()  # Convert back to NumPy array

# %% [markdown]
# The function "train_one_step_new" performs a single training step. It returns the current loss (only needed for debugging purposes). Its parameters are three parallel lists: a minibatch of states, a minibatch of actions, a minibatch of the corresponding TD targets and the discount factor.

# %%
def train_one_step(states, actions, targets, gamma):
  # pass to this function: state1_batch, action_batch, TD_batch
  global model, model2
  state1_batch = torch.cat([torch.from_numpy(s).float() for s in states])
  # state1_batch = torch.tensor(np.array(states), dtype=torch.float32)
  action_batch = torch.LongTensor(actions)
  Q1 = model(state1_batch)
  # print(state1_batch.shape)
  # print(action_batch.shape)
  X = Q1.gather(dim=1,index=action_batch.unsqueeze(dim=1)).squeeze()
  Y = torch.tensor(targets)
  loss = loss_fn(X, Y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss.item()

# %%



