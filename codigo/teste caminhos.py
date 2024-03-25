# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:19:48 2024

@author: joao_
"""

import random

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
        
        
        
    
#%% definiçao de distancia entre arcs em duas maneiras diferentes
# Uma tem como input os arcs da seguinte maneira (('11','10'),('5','12'))
# A outra tem como input os acrs da seguinte maneira (7,32,arc_dict)

def distancia_entre_arcs_nos(arc1,arc2):
    
    node1 = arc1[1]
    node2 = arc2[0]
    
    path = dict_paths_all[node1][node2]
    
    
    
    Custo = shortest_distance_matrix[int(node1)-1,int(node2)-1]
    return path, Custo


        
        
path,custo = distancia_entre_arcs_nos(('11','10'),('5','12')) #teste da funçao  (FUNCTIONA)

print('Path = ',path)
print('Custo = ', custo)
      

    
def distancia_entre_arcs_numerados(arc1, arc2, informaçao_graph):
    
    
    node1 = informaçao_graph[int(arc1)]['node2']
    node2 = informaçao_graph[int(arc2)]['node1']
    
    path = dict_paths_all[node1][node2]
    
    Custo = shortest_distance_matrix[int(node1)-1,int(node2)-1]
    return path, Custo
    
path1,custo1 = distancia_entre_arcs_numerados(41,20,arc_dict) #teste da funçao (FUNCTIONA)
print('Path1 = ',path1)
print('Custo1 = ', custo1)
    
    

#%% criar as matrizes de 1s e 0s para saber em que arcs é necessarioo veiculo passar e em que dias

def transformar_matriz(matrix_day1):
    

    #mete apenas a primeira linha com 0s e 1s, as outras ficam todas a 0    
    for j in range(len(matrix_day1[0])):
        # Check if all elements in the column are 1
        if all(row[j] == 1 for row in matrix_day1):
            # Find the first occurrence of 1 in the column
            found_one = False
            for i in range(len(matrix_day1)):
                if matrix_day1[i][j] == 1 and not found_one:
                    found_one = True
                elif matrix_day1[i][j] == 1 and found_one:
                    matrix_day1[i][j] = 0
    
    # Probability of a 1 passing to another row
    probability = 4/5
    
    # Iterate over each column
    for j in range(len(matrix_day1[0])):
        # Check if the first row has a 1 in this column
        if matrix_day1[0][j] == 1:
            # If yes, randomly decide if it should pass to another row
            if random.random() < probability:
                # Find a random row to pass the 1 to
                new_row_index = random.randint(1, len(matrix_day1) - 1)
                # Move the 1 to the new row and set it to 0 in the first row
                matrix_day1[new_row_index][j] = 1
                matrix_day1[0][j] = 0
 
    array = np.array(matrix_day1, dtype=np.float64)
    return array


matrix_day1 =[]

linhas = numbTrucks
colunas = arcs

#cria uma matriz de 1s e 0s (1 o truck precisa de passar la; 0 nao precisa)
for i in range(linhas):
    row=[]
    for key in arc_dict:
        
        if arc_dict[key]['freq'] == 5:
            element = 1
        elif arc_dict[key]['freq'] == 4:
            
            element = 1
        elif arc_dict[key]['freq'] == 3:
            
            element = 1
        else:
            element = 0
        row.append(element)
    matrix_day1.append(row)
            
matrix_day1 = transformar_matriz(matrix_day1)

matrix_day2 = []
for i in range(linhas):
    row=[]
    for key in arc_dict:
        
        if arc_dict[key]['freq'] == 5:
            element = 1
        elif arc_dict[key]['freq'] == 4:
            
            element = 1
        elif arc_dict[key]['freq'] == 2:
            
            element = 1
        else:
            element = 0
        row.append(element)
    matrix_day2.append(row)
matrix_day2 = transformar_matriz(matrix_day2)

matrix_day3=[]
for i in range(linhas):
    row=[]
    for key in arc_dict:
        
        if arc_dict[key]['freq'] == 5:
            element = 1
        elif arc_dict[key]['freq'] == 3:
            
            element = 1
        
        else:
            element = 0
        row.append(element)
    matrix_day3.append(row)
matrix_day3 = transformar_matriz(matrix_day3)

matrix_day4=[]
for i in range(linhas):
    row=[]
    for key in arc_dict:
        
        if arc_dict[key]['freq'] == 5:
            element = 1
        elif arc_dict[key]['freq'] == 4:
            
            element = 1
        
        else:
            element = 0
        row.append(element)
    matrix_day4.append(row)
matrix_day4 = transformar_matriz(matrix_day4)

matrix_day5=[]
for i in range(linhas):
    row=[]
    for key in arc_dict:
        element=1
       
        row.append(element)
    matrix_day5.append(row)
matrix_day5 = transformar_matriz(matrix_day5)
       
        
