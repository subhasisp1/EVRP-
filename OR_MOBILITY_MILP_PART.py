#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np

# Load Solomon dataset (RC102.csv)
#df = pd.read_csv('R101.csv')
#df = pd.read_csv('RC101_test.csv')
#df = pd.read_csv('RC205_test.csv')
df = pd.read_csv('C106_test.csv')
# Example structure:
# CUST NO., XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE, SERVICE TIME

# Function to calculate Euclidean distance between two points
def calc_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Initialize distance and energy consumption arrays
n_nodes = len(df)
distances = np.zeros((n_nodes, n_nodes))
energy_consumption = np.zeros((n_nodes, n_nodes))


# Randomly assign some nodes as grid points where power can be sold (not the depot)
df['Is_Grid_Node'] = np.random.choice([0, 1], size=n_nodes, p=[0.9, 0.1])  # 10% chance of being a grid node
df.loc[df['CUST NO.'] == 0, 'Is_Grid_Node'] = 0  # Ensure the depot is not a grid node

# Save the modified dataset
df.to_csv('modified_C106_test.csv', index=False)


# In[20]:


df = pd.read_csv('modified_C106_test.csv')
df.describe()


# In[21]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# Load the modified dataset (with time windows included)
df = pd.read_csv('modified_C106_test.csv')

n_nodes = len(df)
vehicles = 12#xmple: 10 EVs
battery_capacity = 100
alpha = 5  # Weight for distance
beta = 10 # Weight for energy consumption
gamma = 4  # Revenue from selling power per unit

# Extract ready time, due date, and service time
ready_time = df['READY TIME'].to_numpy()
due_date = df['DUE DATE'].to_numpy()
service_time = df['SERVICE TIME'].to_numpy()

# Initialize distance and energy consumption arrays
distances = np.zeros((n_nodes, n_nodes))
energy_consumption = np.zeros((n_nodes, n_nodes))

# Function to calculate Euclidean distance
def calc_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Populate the distance and energy consumption matrices
for i in range(n_nodes):
    for j in range(n_nodes):
        distances[i][j] = calc_distance(df['XCOORD.'][i], df['XCOORD.'][j], df['YCOORD.'][i], df['YCOORD.'][j])
        energy_consumption[i][j] = distances[i][j] * 0.5 # Energy consumption per km

# Set grid nodes (some nodes where power can be sold)
is_grid_node = df['Is_Grid_Node'].to_numpy()

# Create Gurobi model
model = gp.Model("EVRP_Multiservice_With_Time_Windows")

# Decision variables
x = model.addVars(n_nodes, n_nodes, vehicles, vtype=GRB.BINARY, name="x")  # Route decisions
y = model.addVars(n_nodes, vehicles, vtype=GRB.BINARY, name="y")  # Power selling decisions at grid nodes
t = model.addVars(n_nodes, vehicles, vtype=GRB.CONTINUOUS, name="t")  # Start time at node i for vehicle k



# Penalize slack in the objective function (higher penalty to discourage violating time windows)
model.setObjective(
    gp.quicksum(alpha * distances[i][j] * x[i,j,k] + beta * energy_consumption[i][j] * x[i,j,k] - gamma * y[j,k]
                for i in range(n_nodes) for j in range(n_nodes) for k in range(vehicles)),
    GRB.MINIMIZE
)


# Constraints
# 1. Each customer must be visited exactly once
for i in range(1, n_nodes):  # Exclude depot (node 0)
    model.addConstr(gp.quicksum(x[i,j,k] for j in range(n_nodes) for k in range(vehicles) if j != i) == 1)

# 2. Flow conservation: if a vehicle arrives at a node, it must leave it
for k in range(vehicles):
    for j in range(1, n_nodes):
        model.addConstr(gp.quicksum(x[i,j,k] for i in range(n_nodes) if i != j) == gp.quicksum(x[j,i,k] for i in range(n_nodes) if i != j))

# 3. Time window constraints
for k in range(vehicles):
    for i in range(n_nodes):
        for j in range(1, n_nodes):  # Exclude depot
            if i != j:
                model.addConstr(t[i, k] + service_time[i] + distances[i][j] <= t[j, k] + (1 - x[i, j, k]) * 10000)

        # Enforce the ready time and due date for each customer
        model.addConstr(t[i, k] >= ready_time[i])
        model.addConstr(t[i, k] <= due_date[i])

# 4. Battery capacity constraint
for k in range(vehicles):
    model.addConstr(gp.quicksum(energy_consumption[i][j] * x[i,j,k] for i in range(n_nodes) for j in range(n_nodes) if i != j) <= battery_capacity)

# 5. Power can only be sold at grid nodes
for j in range(n_nodes):
    if not is_grid_node[j]:
        for k in range(vehicles):
            model.addConstr(y[j,k] == 0)

# 6. If an EV sells power at a node, it must visit that node
for j in range(n_nodes):
    for k in range(vehicles):
        model.addConstr(y[j,k] <= gp.quicksum(x[i,j,k] for i in range(n_nodes) if i != j))

# 7. EVs must start and end at the depot (node 0)
for k in range(vehicles):
    model.addConstr(gp.quicksum(x[0,j,k] for j in range(1, n_nodes)) == 1)  # Leave depot
    model.addConstr(gp.quicksum(x[j,0,k] for j in range(1, n_nodes)) == 1)  # Return to depot
    
model.Params.TimeLimit = 6000  # 600 seconds (10 minutes)   

# Solve the model
model.optimize()

# Extract solution
if model.status == GRB.OPTIMAL:
    solution_x = model.getAttr('x', x)
    solution_y = model.getAttr('x', y)
    solution_t = model.getAttr('x', t)
    print("Optimal solution found")
else:
    print("No optimal solution found")


# In[22]:


# Check the model status to ensure it solved successfully
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
    # Retrieve the objective value (solution accuracy indicator)
    if model.status == GRB.OPTIMAL:
        print(f"Optimal solution found with objective value: {model.ObjVal}")
    else:
        print(f"Best feasible solution found with objective value: {model.ObjVal}")
    
    # Get the computation time
    computation_time = model.Runtime
    print(f"Computation time: {computation_time:.2f} seconds")

    # Get the MIP gap to check solution accuracy
    mip_gap = model.MIPGap * 100  # Convert from fraction to percentage
    print(f"Solution accuracy (MIP gap): {mip_gap:.4f}%")

else:
    print("No feasible solution found.")


# In[ ]:




