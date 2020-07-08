from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2     #if not already done: pip install ortools
from ortools.constraint_solver import pywrapcp
import numpy as np

def create_data_model(distance_matrix):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = np.ndarray.tolist(distance_matrix)  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def tsp_tour_comp(distance_matrix):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(distance_matrix)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    
    tsp_route=[]
    index = routing.Start(0)
    previous_index = index
    index = solution.Value(routing.NextVar(index))
    while not routing.IsEnd(index):
        tsp_route.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        #route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    #index = routing.Start(0)
    #tsp_route.append(manager.IndexToNode(index))
    
    return tsp_route#, route_distance
    
def tsp_tour(tour_school, warehouse, dist):
    if (len(tour_school)<=2): return list(tour_school)
    tour_complete = np.zeros(len(tour_school)+1,dtype=int)
    tour_complete[1:] = np.array(tour_school)
    tour_complete[0]  = warehouse
    
    distance_matrix = dist[np.ix_(tour_complete,tour_complete)]*100000
    tsp_sol = tsp_tour_comp(distance_matrix)
    
    tsp_tour = np.ndarray.tolist(tour_complete[tsp_sol])  #converting indices back from range(len(tour)) to range(M+N)


    
    return tsp_tour
