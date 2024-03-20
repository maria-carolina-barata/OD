# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:43:58 2024

@author: joao_
"""

import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import pulp
import networkx as nx
import matplotlib.pyplot as plt


##############################################################################

# Open the text file
# with open('val16 - Cópia.txt', 'r') as file:
with open('testeProj.txt', 'r') as file:
    
    # Read lines from the file
    lines = file.readlines()

# Initialize a dictionary to store arcs
arc_dict = {}

# Process each line
for line in lines:
    # Split the line by comma delimiter
    values = line.strip().split(', ')
    
    # Create a dictionary to store arc information
    arc_info = {}
    
    # Process each value
    for value in values:
        # Split key and value by '=' delimiter
        key, val = value.split('=')
        # Remove any leading or trailing whitespace from key and value
        key = key.strip()
        val = val.strip()
        # Store key-value pairs in the dictionary
        arc_info[key] = val
    
    # Add the arc information to the arc dictionary
    arc_dict[int(arc_info['arc'])] = arc_info

# custo_total = 0
# for k in range(len(arc_dict)):
#     m = k+1
#     custo_total = float(arc_dict[m]['Trav']) + custo_total
    
    
    
max_value_nodes = float('-inf')

num_arcs = len(arc_dict)

for k in range(len(arc_dict)):
    m_nodes = k+1
    number_nodes = int(arc_dict[m_nodes]['From']) 
    if number_nodes > max_value_nodes:
       # If it is, update the maximum value
       max_value_nodes = number_nodes
       
max_value_days = float('-inf')

for k in range(len(arc_dict)):
    m = k+1
    number = int(arc_dict[m]['freq']) 
    if number > max_value_days:
       # If it is, update the maximum value
       max_value_days = number

days = max_value_days


# Define the problem
prob = pulp.LpProblem("WasteCollection", pulp.LpMinimize)


arcs = []
for k in range(len(arc_dict)):
    m = k+1
    node1 = arc_dict[m]['From']
    node2 = arc_dict[m]['to']
    cost_trav = arc_dict[m]['Trav']
    arcs.append((int(node1),int(node2),int(cost_trav)))
    
arcs_cost =[]

for i in range(num_arcs):
    arcs_cost.append(arcs[i][2])

arcs = arcs[1:]




#%%

G = nx.Graph()  # Create an undirected graph

# Add edges from the list
for edge in arcs:
    node1, node2, distance = edge
    G.add_edge(node1, node2, weight=distance)
    
# Draw the graph
pos = nx.spring_layout(G)  # Positions nodes using Fruchterman-Reingold force-directed algorithm
nx.draw(G, pos, with_labels=True, node_size=100, node_color="skyblue", font_size=12, font_weight="bold")

# Add edge labels (distances)
edge_labels = {(node1, node2): distance for node1, node2, distance in arcs}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Graph Visualization")
plt.show()

#%%







