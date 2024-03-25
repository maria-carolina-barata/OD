import re

# Initialize empty dictionary to store the arc information
arcs = {}

# Open the file and read line by line
with open('gdb1.txt', 'r') as file:
    for line_num, line in enumerate(file, 1): 
        # Check if the line contains the keyword 'arc=' which indicates an arc entry
        if 'arc=' in line:
            # Extraction of necessary information
            arc_number = int(re.search(r'arc=(\d+)', line).group(1))
            cost = int(re.search(r'Trav\s*=\s*(\d+)', line).group(1))
            daily_demand = int(re.search(r'Qty\s*=\s*(-?\d+)', line).group(1))
            frequency = int(re.search(r'freq\s*=\s*(\d+)', line).group(1))

            # Set demand to zero if it's a negative quantity to get all demands positive 
            if daily_demand < 0:
                daily_demand = 1

            # Create a tuple for the From and To nodes (assuming 'From' and 'To' are unique identifiers for arcs)
            from_node = int(re.search(r'From=(\d+)', line).group(1))
            to_node = int(re.search(r'to=(\d+)', line).group(1))
            arc_key = (from_node, to_node)

            # Populate dictionary
            arcs[arc_key] = {'cost': cost, 'daily_demand': daily_demand, 'frequency': frequency}

# Time horizon
num_days = 5

# Vehicle capacity
vehicle_capacity = 15

# Initial number of vehicles -- this may need to be adjusted
num_vehicles = 5

# Initialize the vehicles for each day of the week
vehicles = {
    day: [vehicle_capacity] * num_vehicles for day in range(num_days)
}

# Initialize the service frequency tracking for each arc
arc_service_frequency = {arc: 0 for arc in arcs}


def heuristic_pcarp_weekly(arcs, vehicles, num_days, arc_service_frequency):
    weekly_schedule = {day: [] for day in range(num_days)}

    for day in range(num_days):
        arcs_serviced_today = set()
        arcs_serviced_demand1 = set()
        last_node_visited = {vehicle: 1 for vehicle in range(len(vehicles[day]))}

        for vehicle in range(len(vehicles[day])):
            current_vehicle = vehicle
            visited_arcs = set()
            last_node = 1

            daily_route = [(current_vehicle, 1)]
            last_arc_end = 1

            while len(arcs_serviced_today) < len(arcs) and len(daily_route) < len(arcs) * 2:
                candidate_arcs = [arc_key for arc_key, props in arcs.items() if arc_key not in arcs_serviced_today and arc_key not in visited_arcs and arc_key not in arcs_serviced_demand1 and props['frequency'] > arc_service_frequency[arc_key] and arc_key[0] == last_arc_end]

                if not candidate_arcs:
                    break

                selected_arc = candidate_arcs[0]
                props = arcs[selected_arc]

                daily_route.append((current_vehicle, selected_arc))
                arcs_serviced_today.add(selected_arc)
                visited_arcs.add(selected_arc)
                arc_service_frequency[selected_arc] += 1
                last_arc_end = selected_arc[1]

                if props['daily_demand'] == 1:
                    arcs_serviced_demand1.add(selected_arc)

                current_vehicle = (current_vehicle + 1) % len(vehicles[day])

            daily_route.append((current_vehicle, 1))

            # Ensure that the route starts and ends at the depot node
            if daily_route[0][1] != 1:
                daily_route.insert(0, (current_vehicle, 1))
            if daily_route[-1][1] != 1:
                daily_route.append((current_vehicle, 1))

            # Check if all arcs are visited with required frequency
            for arc_key, props in arcs.items():
                if arc_key not in arcs_serviced_today and props['frequency'] > arc_service_frequency[arc_key]:
                    daily_route.append((current_vehicle, arc_key))
                    arcs_serviced_today.add(arc_key)
                    arc_service_frequency[arc_key] += 1

            weekly_schedule[day].extend(daily_route)

    return weekly_schedule





# Solve the PCARP with frequency and daily demand
weekly_solution = heuristic_pcarp_weekly(arcs, vehicles, num_days, arc_service_frequency)


# Display the solution without depot nodes at the beginning of routes
for day, daily_routes in weekly_solution.items():
    print(f"Day {day}:")
    for vehicle in range(len(vehicles[day])):
        vehicle_route = []  # Initialize the vehicle route list for each vehicle
        for v, arc in daily_routes:
            if v == vehicle:
                if arc != 1:  # Exclude the depot node (1) from the route
                    vehicle_route.append(f"({arc})")

        # Only print the route if it contains valid arcs
        if len(vehicle_route) > 1:
            print(f"  Vehicle {vehicle} services: {' '.join(vehicle_route)}")



