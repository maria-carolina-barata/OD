import re
import networkx as nx
import matplotlib.pyplot as plt

# Initialize an empty dictionary to store arc information
arcs = {}

# Open the file and read line by line
with open('gdb9.txt', 'r') as file:
    for line_num, line in enumerate(file, 1):  # Add line number for debugging
        # Check if the line contains the keyword 'arc=' which indicates an arc entry
        if 'arc=' in line:
            # Use regular expressions to find the necessary information
            arc_number = int(re.search(r'arc=(\d+)', line).group(1))
            cost = int(re.search(r'Trav\s*=\s*(\d+)', line).group(1))
            daily_demand = int(re.search(r'Qty\s*=\s*(-?\d+)', line).group(1))
            frequency = int(re.search(r'freq\s*=\s*(\d+)', line).group(1))

            # Create a tuple for the From and To nodes (assuming 'From' and 'To' are unique identifiers for arcs)
            from_node = int(re.search(r'From=(\d+)', line).group(1))
            to_node = int(re.search(r'to=(\d+)', line).group(1))
            arc_key = (from_node, to_node)

            # Populate the arcs dictionary
            arcs[arc_key] = {'cost': cost, 'daily_demand': daily_demand, 'frequency': frequency}

# Print every arc with frequency 5
for arc, info in arcs.items():
    if info['frequency'] == 1:
        from_node, to_node = arc
        print(f"Arc: {arc}, Cost: {info['cost']}, Daily Demand: {info['daily_demand']}, Frequency: {info['frequency']}")

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph based on the arcs information
for arc, info in arcs.items():
    from_node, to_node = arc
    G.add_edge(from_node, to_node, cost=info['cost'], daily_demand=info['daily_demand'], frequency=info['frequency'])

# Draw the network graph
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'cost')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# Display the graph
plt.title("Network Graph")
plt.axis('off')
plt.show()






