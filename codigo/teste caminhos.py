# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:19:48 2024

@author: joao_
"""


import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import pulp
import heapq

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import networkx as nx
import matplotlib.pyplot as plt

def dijkstra(graph, start):
    # Initialize distances to all nodes as infinity
    distances = {node: float('inf') for node in graph}
    # Set distance from start node to itself as 0
    distances[start] = 0
    # Priority queue to store nodes and their tentative distances
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # If current distance is greater than known shortest distance, ignore
        if current_distance > distances[current_node]:
            continue

        # Iterate through neighbors of current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # If new distance is shorter, update
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Open the text file
# with open('val16.txt', 'r') as file:
with open('gdb1.txt', 'r') as file:
    # Read lines from the file
    lines = file.readlines()

# Initialize a dictionary to store arcs
arc_dict = {}



numbNodes = int(lines[1].split('=')[1])
numbTrucks = int(lines[5].split('=')[1])
TruckCapacity = int(lines[6].split('=')[1])


graph_arcs={}

# Process each line
for line in lines[10:]:
    # Split the line by comma delimiter
    values = line.split(' ')
    
    # Create a dictionary to store arc information
    arc_info = {}
    numbArc = values[1].split('=')[1]
    numbArc= numbArc.split(',')[0]
    node1 = values[2].split('=')[1]
    node1 = node1.split(',')[0]
    node2 = values[3].split('=')[1]
    node2 = node2.split(',')[0]
    travCost = values[5].split('=')[1]
    travCost = travCost.split(',')[0]
    freq = int(values[10][0])
    
    inv_and_succ=values[8].split('=')
    inv = inv_and_succ[1].split(',')[0]
    
    succ=inv_and_succ[2].split(',')
    succ=succ[:-1]
    
    
        
        
    
    arc_info['arc']=str(numbArc)
    arc_info['node1']=node1
    arc_info['node2']=node2
    arc_info['Trav']=int(travCost)
    arc_info['freq']=freq
    arc_info['inv']=inv
    arc_info['succ']=succ
    
    
   
    # Add the arc information to the arc dictionary
    arc_dict[int(arc_info['arc'])] = arc_info

arcs_nodes={}###########
for arcs in arc_dict: 
    arc_succ={}
    arcs_nodes_succ={}#########
    
    node1 = arc_dict[arcs]['node1']
    node2 = arc_dict[arcs]['node2']
    trav =  arc_dict[arcs]['Trav']
    
    arcs_nodes[(f'{node1}',f'{node2}')]=trav
    
    for arcSucc in  arc_dict[arcs]['succ']:
        arcSucc=int(arcSucc)
        arc_succ[arcSucc]=arc_dict[arcSucc]['Trav']
       
    
    # arcs_nodes[node1]=arcs_nodes_succ
    graph_arcs[f'{arcs}']=arc_succ
    


    

# Convert numerical keys to strings for both the main dictionary and nested dictionaries
new_graph_arcs = {str(key): {str(k): v for k, v in value.items()} for key, value in graph_arcs.items()}


graph_with_nodes={}

for key, inner_dict in graph_arcs.items():
   
    key = int(key)
    node_ini = arc_dict[key]['node2']
    node_ini = int(node_ini)
    new_inner_dict={}
    for inner_key, trav in inner_dict.items():
        
        
        inner_key = int(inner_key)
        node_final=arc_dict[inner_key]['node2']
        
        new_inner_dict[node_final]=trav
    
    graph_with_nodes[node_ini]=new_inner_dict


# Convert keys from numbers to strings
new_graph_with_nodes = {str(key): value for key, value in graph_with_nodes.items()}


for key in sorted(new_graph_with_nodes, key=lambda x: int(x)):
    start_node = f'{key}'
    shortest_distances = dijkstra(new_graph_with_nodes, start_node)
    # print("Shortest distances from node", start_node + ":")
    # print(shortest_distances)



# Get the unique nodes in the graph
nodes = sorted(new_graph_with_nodes, key=lambda x: int(x))

# Initialize a matrix to store the shortest distances
num_nodes = len(nodes)
shortest_distance_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float)

# Compute the shortest distances between all pairs of nodes
for i, start_node in enumerate(nodes):
    shortest_distances = dijkstra(new_graph_with_nodes, start_node)
    for j, end_node in enumerate(nodes):
        shortest_distance_matrix[i, j] = shortest_distances[end_node]

#%%   Graph where it outputs the shortest path





def dijkstra(graph, start):
    # Initialize distances to all nodes as infinity
    distances = {node: float('inf') for node in graph}
    # Set distance from start node to itself as 0
    distances[start] = 0
    # Priority queue to store nodes and their tentative distances
    pq = [(0, start)]
    # Dictionary to store predecessors
    predecessors = {}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # If current distance is greater than known shortest distance, ignore
        if current_distance > distances[current_node]:
            continue

        # Iterate through neighbors of current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # If new distance is shorter, update
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors

def shortest_path(graph, start, end):
    distances, predecessors = dijkstra(graph, start)
    path = []
    current_node = end

    while current_node != start:
        path.append(current_node)
        current_node = predecessors[current_node]

    path.append(start)
    path.reverse()
    return path

# Example graph


start_node = '1'
end_node = '3'
shortest_distances, predecessors = dijkstra(new_graph_with_nodes, start_node)
shortest_path_nodes = shortest_path(new_graph_with_nodes, start_node, end_node)
print("Shortest path from", start_node, "to", end_node + ":")
print(shortest_path_nodes)



dict_paths_all={}

for key_ini in sorted(new_graph_with_nodes, key=lambda x: int(x)):
    start_node = f'{key_ini}'
    dict_path_in = {}
    for key_end in sorted(new_graph_with_nodes, key=lambda x: int(x)):
        end_node= f'{key_end}'
        shortest_distances, predecessors = dijkstra(new_graph_with_nodes, start_node)
        shortest_path_nodes = shortest_path(new_graph_with_nodes, start_node, end_node)
        dict_path_in[end_node]= shortest_path_nodes
    dict_paths_all[start_node] =   dict_path_in
        
        
        
        
        
