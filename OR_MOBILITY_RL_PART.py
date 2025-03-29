#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym


# In[2]:


import pandas as pd
import numpy as np

# Load your dataset (replace 'RC103.csv' with the actual file path)
df = pd.read_csv("modified_RC203_test.csv")

# Display the dataset structure to understand its columns
print(df.head())

# Extract relevant columns from the dataframe
customer_ids = df['CUST NO.'].values
coordinates = df[['XCOORD.', 'YCOORD.']].values  # Assuming coordinates are X and Y positions of the customers
demands = df['DEMAND'].values
ready_times = df['READY TIME'].values
due_dates = df['DUE DATE'].values
service_times = df['SERVICE TIME'].values
is_grid_node = df['Is_Grid_Node'].values.astype(bool)  # Binary indicating if it's a grid node


# In[7]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


class EVRoutingEnvironment:
    def __init__(self, df, grid_nodes, energy_sell_rate, energy_cost_rate, battery_capacity):
        # Initialize with the dataset and parameters
        self.df = df
        self.num_nodes = len(df)  # Number of customers
        self.distance_matrix = self._generate_distance_matrix()  # Generate distance matrix

        # Extract relevant columns from the dataframe
        self.ready_times = df['READY TIME'].values
        self.due_dates = df['DUE DATE'].values
        self.service_times = df['SERVICE TIME'].values
        self.demands = df['DEMAND'].values
        self.grid_nodes = grid_nodes
        self.energy_sell_rate = energy_sell_rate
        self.energy_cost_rate = energy_cost_rate
        self.battery_capacity = battery_capacity

        # Initialize state
        self.current_state = [0, 0, 0, battery_capacity]  # [current_node, demand, time, battery_level]
        self.visited_nodes = []  # To track visited nodes
        self.current_charge = battery_capacity

    def _generate_distance_matrix(self):
        # Generate a distance matrix based on Euclidean distance (replace with your actual method if needed)
        num_nodes = len(self.df)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                x1, y1 = self.df.iloc[i][['XCOORD.', 'YCOORD.']].values
                x2, y2 = self.df.iloc[j][['XCOORD.', 'YCOORD.']].values
                distance_matrix[i][j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance_matrix

    def reset(self):
        # Reset the environment at the start of each episode
        self.visited_nodes = []  # Clear visited nodes
        self.current_state = [0, 0, 0, self.battery_capacity]  # Reset state
        self.current_charge = self.battery_capacity
        return np.array(self.current_state)

    def step(self, action):
        # Take action (move to next node) and update the state
        next_node = action
        done = False
        reward = 0

        current_node = self.current_state[0]
        distance = self.distance_matrix[current_node][next_node]
        energy_consumed = distance  # Assuming energy consumption is proportional to distance

        # If the next node is a grid node, sell energy
        energy_sold = 0
        if self.df.iloc[next_node]['Is_Grid_Node']:
            energy_sold = min(self.current_charge, 50)  # Sell up to 50 units of energy
            self.current_charge -= energy_sold

        # Update time and battery usage
        travel_time = distance
        self.current_state[2] += travel_time  # Update the current time

        # Check if the EV arrives too late
        arrive_late = 1 if self.current_state[2] > self.due_dates[next_node] else 0

        # Calculate reward based on distance, energy consumption, and potential revenue from energy sale
        reward = (
            -distance * 7 * self.energy_cost_rate  # Penalty for distance
            - energy_consumed * 7 * self.energy_cost_rate  # Penalty for energy consumption
            + 0.00001 * energy_sold * self.energy_sell_rate  # Reward for selling energy
            - 300 * (arrive_late)  # Penalty for arriving late
        )


        # Update state
        self.current_state = [next_node, self.demands[next_node], self.current_state[2], self.current_charge]

        # Mark the node as visited
        self.visited_nodes.append(next_node)
        if len(self.visited_nodes) == self.num_nodes:  # If all nodes have been visited, episode is done
            done = True

        return np.array(self.current_state), reward, done, {}



# In[4]:


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration factor
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory = []
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Linear output for Q-values
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        q_values = self.model.predict(state, verbose=0)  # Get Q-values for each action
        return np.argmax(q_values[0])  # Action with the highest Q-value (exploitation)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = -reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# In[5]:


def train_dqn(env, agent, episodes=20, batch_size=8):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_cost = 0
        for time in range(500):  # Maximum steps per episode
            action = agent.act(state)  # Choose an action
            next_state, reward, done, _ = env.step(action)  # Take the action and get the reward
            next_state = np.reshape(next_state, [1, agent.state_size])

            # Store the experience in memory
            agent.memory.append((state, action, reward, next_state, done))

            # Learn from the experience
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            state = next_state
            total_cost += reward

            if done:
                print(f"Episode {e+1}/{episodes}, Total Cost: {-total_cost}")
                break

# Define the grid nodes (indices of grid nodes in the dataset)
grid_nodes = [i for i, is_grid in enumerate(is_grid_node) if is_grid]

# Energy selling parameters
energy_sell_rate = 0.00001  # Rate for selling energy
energy_cost_rate = 30  # Rate for energy consumed
battery_capacity = 50  # Example battery capacity

# Initialize the environment and agent
env = EVRoutingEnvironment(df, grid_nodes, energy_sell_rate, energy_cost_rate, battery_capacity)
state_size = len(env.current_state)  # State size based on the state vector
action_size = len(df)  # Action size based on the number of nodes
agent = DQNAgent(state_size, action_size)


# In[6]:


train_dqn(env, agent, episodes=20, batch_size=8)


# In[7]:


from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

agent.model.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.005)
)
agent.model.save("trained_dqn_model.h5")



# In[8]:


from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

trained_model = load_model("trained_dqn_model.h5", custom_objects={"MeanSquaredError": MeanSquaredError})



# In[9]:


import time
import numpy as np

def solve_route_with_time(env, trained_model):
    state = env.reset()  # Reset the environment
    state = np.reshape(state, [1, trained_model.input_shape[1]])  # Prepare state for the model
    done = False
    route = []  # To store the sequence of visited nodes
    total_cost = 0  # To track the total cost

    start_time = time.time()  # Start timing

    while not done:
        # Predict the Q-values for the current state and choose the action with max Q-value
        q_values = trained_model.predict(state, verbose=0)
        action = np.argmax(q_values[0])

        # Perform the action and get the next state and reward
        next_state, reward, done, _ = env.step(action)
        route.append(action)
        total_cost += reward  # Accumulate reward (negative of cost)

        # Prepare the next state for the model
        state = np.reshape(next_state, [1, trained_model.input_shape[1]])

    end_time = time.time()  # End timing
    computation_time = end_time - start_time  # Calculate total time

    return route, -total_cost, computation_time  # Return route, cost, and computation time


# In[10]:


route, total_cost, computation_time = solve_route_with_time(env, trained_model)
print("Optimal Route:", route)
print("Total Cost:", total_cost)
print(f"Computation Time: {computation_time:.4f} seconds")



# In[ ]:




