# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:32:21 2024

@author: joao_
"""

import re
import networkx as nx
from dataclasses import dataclass
import math
import random
import numpy as np


@dataclass
class Data:
    name: str
    nb_nodes: int
    nb_arcs: int
    nb_tasks: int
    nb_non_req: int
    nb_trucks: int
    truck_capacity: int
    total_cost: int
    depot: int
    graph: nx.DiGraph


def add_arc(G: nx.DiGraph, line: str) -> nx.DiGraph:
    arc = int(re.search(r"arc=(\d+)", line).group(1))
    from_node = int(re.search(r"From=(\d+)", line).group(1))
    to_node = int(re.search(r"to=(\d+)", line).group(1))
    demand = int(re.search(r"Qty=(-?\d+)", line).group(1))
    cost_traversing = int(re.search(r"Trav=(-?\d+)", line).group(1))
    cost_collecting = int(re.search(r"Col= (-?\d+)", line).group(1))
    opposed_arc = int(re.search(r"inv=(\d+)", line).group(1))
    successor_arcs = list(map(int, re.search(r"succ=(.*);", line).group(1).split(",")))
    frequency = int(re.search(r"freq= (\d+)", line).group(1))

    # Check if node exists, if not create it
    if not G.has_node(from_node):
        G.add_node(from_node)
    if not G.has_node(to_node):
        G.add_node(to_node)

    # Removes the arc if it is the same as the opposed arc
    if from_node != to_node:
        G.add_edge(
            from_node,
            to_node,
            arc=arc,
            demand=demand,
            cost_traversing=cost_traversing,
            cost_collecting=cost_collecting,
            opposed_arc=opposed_arc,
            successor_arcs=successor_arcs,
            frequency=frequency,
        )

    return G


def import_data(paht : str) -> Data:

    G = nx.DiGraph()

    with open(paht, "r") as file:
        lines = file.readlines()

        for line in lines:
            if re.match(r"^Name=", line):
                name = re.search(r"=(.*)", line).group(1).strip()
            elif re.match(r"^nbnodes=", line):
                nb_nodes = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^nbarcs=", line):
                nb_arcs = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^nbtasks=", line):
                nb_tasks = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^nbnon_req=", line):
                nb_non_req = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^nbtrucks=", line):
                nb_trucks = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^capacity=", line):
                capacity = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^total_cost=", line):
                total_cost = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^Depot=", line):
                depot = int(re.search(r"=(.*)", line).group(1).strip())
            elif re.match(r"^ arc", line):
                G = add_arc(G, line)

    return Data(
        name=name,
        nb_nodes=nb_nodes,
        nb_arcs=nb_arcs,
        nb_tasks=nb_tasks,
        nb_non_req=nb_non_req,
        nb_trucks=nb_trucks,
        truck_capacity=capacity,
        total_cost=total_cost,
        depot=depot,
        graph=G,
    )

#%%
data = import_data(r"C:/Users/joao_/Desktop/Otimização e Decisão/OD/codigo/gdb1_original.txt")
G = data.graph

data


#%%
print(data.graph)

#%%

def plot_graph(G: nx.DiGraph, path: list | None = None):
    options = {
        "font_size": 24,
        "node_size": 1500,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 3,
        "width": 3,
        "with_labels": True,
    }

    layers = {
        0: [1],
        1: [2, 7, 12],
        2: [10, 4],
        3: [9, 8, 6],
        4: [11, 5, 3],
    }

    pos = {}
    width = 1.0 / (len(layers) - 1)
    for i, (layer, nodes) in enumerate(layers.items()):
        height = 1.0 / (len(nodes) + 1)
        for j, node in enumerate(nodes):
            pos[node] = (i * width, (j + 1) * height)

    nx.draw(G, pos, **options)

    if path is not None:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="r", width=5)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="lightblue", node_size=2000)


plot_graph(G)
#%%

class SimulatedAnnealing:
    data: Data = None 
    current_solution = None
    current_cost = None
    big_M = None
    max_path_length = None

    def __init__(
        self,
        data: Data,
        temperature=1.0, #initial value to control the likelihood of accepting a worse solution
        min_temperature=0.001, #minimum likelihood to control stopping criterion
        alpha=0.9, #rate at which the likelihood decreases
        big_M=1e5, 
        max_path_length=16,
    ):
        self.data = data
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.alpha = alpha
        self.big_M = big_M
        self.max_path_length = max_path_length

    #start with a random route
    def _generate_random_route(self, soft_max_length: int = None):
        start_node = self.data.depot #start at depot node
        if soft_max_length is None: 
            soft_max_length = math.ceil(self.data.nb_arcs / self.data.nb_trucks) # the maximum should be a situation where all streets are equally divided by the trucks
        route = [start_node]

        while True:
            next_node = random.choice(list(self.data.graph[route[-1]])) # random populate a route between the starting and finishing node
            route.append(next_node)

            if next_node == self.data.depot: 
                break

            if len(route) >= soft_max_length:
                route.extend(nx.shortest_path(self.data.graph, route[-1], start_node)[1:])
                break

        return route
    
    # do the same procedure for all trucks in all days to have initial solution
    def _initial_solution(self):
        return [{n: self._generate_random_route() for n in range(data.nb_trucks)} for _ in range(5)] 
    def _add_node(self, route: list[int]):
        # select random node in the route to add a node after
        selected_idx = random.randint(0, len(route) - 2)
        selected_node = route[selected_idx]
        next_node = route[selected_idx + 1]

        # TODO: Add no turn back logic
        # Find the shortest path from the possible new node to the next node
        shortest_path_to_next_node = {
            node: nx.shortest_path(G, node, next_node)
            for node in G[selected_node]
            if node != next_node
        }

        # Other options for selecting the new node
        # new_node_idx = min(
        #     shortest_path_to_next_node,
        #     # key=lambda x: len(shortest_path_to_next_node[x]),
        #     # key=lambda x: sum(
        #     #     G[i][j]["cost_traversing"]
        #     #     for i, j in zip(
        #     #         shortest_path_to_next_node[x],
        #     #         shortest_path_to_next_node[x][1:],
        #     #     )
        #     # ),
        # )
        new_node_idx = random.choice(list(shortest_path_to_next_node.keys()))

        return (
            route[: selected_idx + 1]
            + shortest_path_to_next_node[new_node_idx][:-1]
            + route[selected_idx + 1 :]
        )

    def _remove_node(self, route: list[int]):
        # Remove a node from the route
        pass

    def _swap_nodes(self, route: list[int]):
        # Swap two nodes in the route
        pass

    def _neighbour(self, solution: list[dict]):
        # Generate a neighbour solution by making a small change to the current solution
        # For routing, this could involve swapping two streets in the route, for example
        random_number = random.random()
        day = random.randint(0, len(solution) - 1)
        truck = random.choice(list(solution[day].keys()))
        route = solution[day][truck]

        if len(route) <= 2:
            updated_route = self._add_node(route)
        elif len(route) >= self.max_path_length:
            updated_route = self._remove_node(route)
        elif random_number < 0.2 and len(route) > self.max_path_length:
            updated_route = self._remove_node(route)

        elif random_number < 0.4 and len(route) < self.max_path_length:
            updated_route = self._add_node(route)

        else:
            updated_route = self._swap_nodes(route)

        solution[day][truck] = updated_route

        return solution

    def _cost(self, solution: list[dict]) -> tuple[float, dict]:
        # Collection is evaluated on a first-come-first-serve basis
        all_arcs = set(data.graph.edges)
        # all_arcs.remove((data.depot, data.depot))

        total_traversing_cost = 0
        total_collection_cost = 0

        rubbish = {arc: 0 for arc in all_arcs}
        sol_freq = {arc: 0 for arc in all_arcs}

        for day in solution:
            # Add 1 rubbish to each arc at the beginning of the day
            for arc in all_arcs:
                rubbish[arc] += 1

            arcs_visited_current_day = set()

            for route in day.values():
                remaining_truck_capacity = self.data.truck_capacity

                for i, j in zip(route, route[1:]):
                    arcs_visited_current_day.add((i, j))

                    decrease = min(remaining_truck_capacity, rubbish[(i, j)])

                    remaining_truck_capacity -= decrease
                    rubbish[(i, j)] -= decrease

                    total_traversing_cost += self.data.graph[i][j]["cost_traversing"]
                    total_collection_cost += self.data.graph[i][j]["cost_collecting"] * decrease

            for arc in arcs_visited_current_day:
                sol_freq[arc] += 1

        # Penalty for violating the freq constraint
        freq_penalty = 0
        for i, j in all_arcs:
            freq_penalty += (
                max(0, sol_freq[(i, j)] - self.data.graph[i][j]["frequency"]) * self.big_M
            )

        # Penalty for uncollected rubbish
        uncollected_penalty = 0
        for i, j in all_arcs:
            uncollected_penalty += rubbish[(i, j)] * self.big_M

        total_cost = (
            total_traversing_cost + total_collection_cost + freq_penalty + uncollected_penalty
        )

        return total_cost, {
            "total_traversing_cost": total_traversing_cost,
            "total_collection_cost": total_collection_cost,
            "freq_penalty": freq_penalty,
            "uncollected_penalty": uncollected_penalty,
        }

    def _accept_probability(self, old_cost: float, new_cost: float, temperature: float):
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / temperature)

    def solve(self) -> tuple[list[dict], float]:
        self.current_solution = self._initial_solution()
        self.current_cost = self._cost(self.current_solution)

        print(f"Initial solution cost: {self.current_cost}\nStarting annealing...")

        i = 0
            #while self.temperature > self.min_temperature:
            #new_solution = self._neighbour(self.current_solution) """
            #new_cost = self._cost(new_solution)
            #if (
                #self._accept_probability(self.current_cost, new_cost, self.temperature)
                #> random.random()
            #):
                #self.current_solution = new_solution
                #self.current_cost = new_cost

            #self.temperature *= self.alpha

            #i += 1
            #if i % 100 == 0:
                #print(f"Iteration {i} - Current solution cost: {self.current_cost}")

        print(f"Solver finished in {i} iterations\nBest solution cost: {self.current_cost}")

        return self.current_solution, self.current_cost

    def pretty_print(self):
        print(f"Best solution cost: {self.current_cost}")
        print("Best solution")
        for day, routes in enumerate(self.current_solution):
            print(f"Day {day}")
            for truck, route in routes.items():
                print(f"Truck {truck}: {route}")


prob = SimulatedAnnealing(data)    


#%%
# prob.solution
# plot_graph(G, prob._generate_random_route())

best_solution, best_cost = prob.solve()
prob.pretty_print()

#%%
day = 4
truck = 4
plot_graph(G, best_solution[day][truck])




