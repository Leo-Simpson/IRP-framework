import numpy as np
from tsp_solver.greedy import solve_tsp     #requires pip install tsp_solver2

def tsp_tour(tour,solution):

                
    tour_temp = np.array([tour[0]] + tour)    #The tsp_solver can only solve for different start and endpoints, so I add the WH a second time.
    name_warehouses, name_schools = np.array(solution.name_warehouses), np.array(solution.name_schools)
    tour_in_names = np.concatenate([name_warehouses[tour_temp[:2]], name_schools[tour_temp[2:]]])
    distance_matrix = solution.problem.D.loc[tour_in_names, tour_in_names].values

    tsp_sol = solve_tsp(distance_matrix, endpoints = (0,1))            #the solver with the WH as start and endpoint
    
    tsp_tour = tour_temp[tsp_sol]           #converting indices back from range(len(tour)) to range(M+N)
    return tsp_tour
