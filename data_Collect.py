# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:43:58 2024

@author: joao_
"""

import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import pulp


##############################################################################

# Open the text file
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
    
#%%
# C_trav = [] # cost associated with traversing an arc
# for k in range(len(arc_dict)):
#     m = k+1
#     C_trav.append(arc_dict[m]['Trav'])


# C_collect =[] #(node j, cost associated with collecting in node j)
# for k in range(len(arcs)):
#     m = k+1
#     node2 = arc_dict[m]['to']
#     C_collect.append((node2,arc_dict[m]['Col']))


