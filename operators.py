import numpy as np
from OR_tools_solve_tsp import tsp_tour
import random
import sys

def rand_remove_rho(solution, rho):
    for i in range( min(rho,np.sum(solution.Y)) ):  
        t,n,k,m = random.choice(np.transpose(np.nonzero(solution.Y)))
        solution.Y[t,n,k,m] = 0
        tour = np.array(solution.r[t][n][k])
        solution.r[t][n][k] = np.ndarray.tolist(tour[tour != m + solution.N])


def remove_worst_rho(solution, rho):
    for i in range( min(rho,np.sum(solution.Y)) ):   
        Y_flat = solution.Y.reshape(-1)
        a_flat = solution.a.reshape(-1)
        argmax = np.argmax(a_flat[Y_flat == 1])
        choice = np.where(Y_flat == 1)[0][argmax]
        Y_flat[choice] = 0
        a_flat[choice] = 0
        t, rest = np.divmod(choice, solution.N*solution.K*solution.M)
        n, rest = np.divmod(rest, solution.K*solution.M)
        k,m = np.divmod(rest, solution.M)
        if m+solution.N in solution.r[t][n][k]: solution.r[t][n][k].remove(m+solution.N)
        #tour = np.array(solution.r[t][n][k])
        #solution.r[t][n][k] = np.ndarray.tolist(tour[tour != m + solution.N])
        solution.compute_school_remove_dist(t,n,k)

def shaw_removal_route_based(solution, rho):
    t,n,k,m = random.choice(np.transpose(np.nonzero(solution.Y)))
    route = np.array(solution.r[t][n][k])
    if len(route) > 2:
        schools = route[np.where(route != m + solution.N)[0]]
        dist_from_m = solution.problem.D[np.ix_([m + solution.N],route)][0]
        min_dist_from_m = np.min(solution.problem.D[np.ix_([m + solution.N],schools)][0])
        to_remove = route[np.where(dist_from_m <= 2*min_dist_from_m)[0]] - solution.N 
        solution.Y[t,n,k,to_remove] = 0
        solution.r[t][n][k] = tsp_tour(np.setdiff1d(route, to_remove + solution.N), n, solution.problem.D)
    else:
        solution.Y[t,n,k,:] = 0
        solution.r[t][n][k] = []

def shaw_removal_greedy(solution, rho):
    t,n,k,m = random.choice(np.transpose(np.nonzero(solution.Y)))
    route = np.array(solution.r[t][n][k])
    if len(route) > 2:
        schools = route[np.where(route != m + solution.N)[0]]
        dist_from_m = solution.problem.D[np.ix_([m + solution.N],route)][0]
        dist = solution.problem.D[np.ix_(route,route)]
        min_dist = np.min(dist[dist>0])
        to_remove = route[np.where(dist_from_m <= 2*min_dist)[0]] - solution.N 
        solution.Y[t,n,k,to_remove] = 0
        solution.r[t][n][k] = tsp_tour(np.setdiff1d(route, to_remove + solution.N), n, solution.problem.D)
    else:
        solution.Y[t,n,k,:] = 0
        solution.r[t][n][k] = []

def avoid_consecutive_visits(solution, rho):
    for t in range(solution.T-1):
        time_schools = np.sum(solution.Y[[t, t+1],:,:,:], axis = (1,2))
        index = np.where(time_schools[1,:] + time_schools[0,:] > 1)[0]
        change = np.transpose(np.where(np.sum(solution.Y[:,:,:,index][t+1], axis = 2) > 0))
        solution.Y[t+1,:,:,index] = 0
        for n,k in change:
            schools = np.nonzero(solution.Y[t+1,n,k,:])[0]
            solution.r[t+1][n][k] = tsp_tour(schools  + solution.N, n, solution.problem.D)



def empty_one_period(solution, rho):
    period = np.random.randint(solution.T)
    solution.Y[period,:,:,:] = 0
    solution.r[period] = [ [[]]*solution.K]*solution.N

def empty_one_vehicle(solution, rho):
    warehouse = np.random.randint(solution.N)
    vehicle = np.random.randint(solution.K)
    solution.Y[:,warehouse,vehicle,:] = 0
    for t in range(solution.T):
        solution.r[t][warehouse][vehicle] = []

def empty_one_plant(solution, rho):
    warehouse = np.random.randint(solution.N)
    solution.Y[:,warehouse,:,:] = 0
    for t in range(solution.T):
        for k in range(solution.K):
            solution.r[t][warehouse][k] = []
    
def furthest_customer(solution, rho):
    candidates = np.sum(solution.Y, axis = 3)
    for i in range( min(rho,np.sum(candidates)) ):
        t,n,k = random.choice(np.transpose(np.nonzero(candidates)))
        route = solution.r[t][n][k]
        furthest_cust_index = np.argmax(solution.problem.D[np.ix_([n],route)][0])
        solution.Y[t,n,k,route[furthest_cust_index] - solution.N] = 0
        route.pop(furthest_cust_index)
        candidates[t,n,k] -= 1
    
def rand_insert_rho(solution, rho):    
    candidates = ~np.any(solution.Y, axis = (1,2))  # ~ is the negation of a boolean array
    for i in range( min(rho,np.sum(candidates)) ):
        t, m = random.choice(np.transpose(np.nonzero(candidates)))
        candidates[t,m] -= 1
        n, k = random.choice(np.nonzero(solution.Cl[:,m])[0]), np.random.randint(solution.K)
        solution.Y[t,n,k,m] = 1
        solution.r[t][n][k],_ = solution.cheapest_school_insert(t,n,k,m)
    


def assign_to_nearest_plant(solution, rho):
    candidates = ~np.any(solution.Y, axis = (1,2))  # ~ is the negation of a boolean array
    for i in range( min(rho,np.sum(candidates)) ):
        t, m = random.choice(np.transpose(np.nonzero(candidates)))
        allowed_plants = [i for i in range(solution.N) if solution.Cl[i,m] == 1]
        nearest_plant = allowed_plants[np.argmin(solution.problem.D[np.ix_([m + solution.N],allowed_plants)][0])]
        candidates[t,m] = False
        route, cost = solution.cheapest_school_insert(t,nearest_plant,0,m)
        index = 0
        for k in range(1, solution.K):
            route_temp, cost_temp = solution.cheapest_school_insert(t,nearest_plant,k,m)
            if cost_temp < cost:
                route = route_temp
                index = k
        solution.Y[t,nearest_plant,index,m] = 1
        solution.r[t][nearest_plant][index] = route
    
def insert_best_rho(solution, rho):
    candidates = ~np.any(solution.Y, axis = (1,2))   # ~ is the negation of a boolean array
    eliminate = np.transpose(np.where(np.any(solution.Y, axis = (1,2))))
    for t,m in eliminate:
        solution.b[t,:,:,m] = sys.maxsize
    for i in range( min(rho,np.sum(candidates)) ):
        b_flat = solution.b.reshape(-1)
        Y_flat = solution.Y.reshape(-1)
        Cl_flat = solution.Cl_shaped_like_Y().reshape(-1)
        choice = np.where(Cl_flat - Y_flat > 0)[0][np.argmin(b_flat[Cl_flat - Y_flat>0])]
        Y_flat[choice] = 1
        t, rest = np.divmod(choice, solution.N*solution.K*solution.M)
        n, rest = np.divmod(rest, solution.K*solution.M)
        k, m = np.divmod(rest, solution.M)
        #solution.r[t][n][k],_ = solution.cheapest_school_insert(t,n,k,m)
        tour_school = np.nonzero(solution.Y[t,n,k,:])[0] + solution.N 
        solution.r[t][n][k] = tsp_tour(tour_school, n, solution.problem.D)
        solution.compute_school_insert_dist(t,n,k)
        for m_temp in range(solution.M):
            if np.any(solution.Y, axis = (1,2))[t,m_temp]:
                solution.b[t,n,k,m_temp] = sys.maxsize
        solution.b[t,:,:,m] = sys.maxsize
    solution.compute_a_and_b()
        
    
def shaw_insertion(solution, rho): 
    period = np.random.randint(solution.T)
    not_served = np.where(np.sum(solution.Y[period,:,:,:], axis = (0,1)) == 0)[0]
    if len(not_served) > 0:
        (index, choice) = random.choice(list(enumerate(not_served)))
        dist_to_all = solution.problem.D[np.ix_([choice + solution.N],[m  + solution.N for m in range(solution.M) if m != choice])][0]
        rest_not_served = np.delete(not_served, index)
        dist_to_not_served = solution.problem.D[np.ix_([choice + solution.N],rest_not_served + solution.N)][0]
        close = rest_not_served[dist_to_not_served <= 2*np.min(dist_to_all)]
        closest_warehouse = np.argmin(solution.problem.D[np.ix_([choice + solution.N],[i for i in range(solution.N)])][0])
        close_reachable = [m for m in close[solution.Cl[closest_warehouse, close] == 1]]
        route = tsp_tour(close_reachable + solution.r[period][closest_warehouse][0],closest_warehouse,solution.problem.D)
        costs = solution.compute_route_dist(route, closest_warehouse)
        index = 0
        for k in range(1, solution.K):
            route_temp = tsp_tour(close_reachable + solution.r[period][closest_warehouse][k],closest_warehouse,solution.problem.D)
            costs_temp = solution.compute_route_dist(route_temp, closest_warehouse)
            if costs_temp < costs:
                route = route_temp
                index = k
        solution.Y[period,closest_warehouse,index, close_reachable] = 1
        solution.r[period][closest_warehouse][index] = route
    
def swap_rho_cust_intra_routes(solution, rho):
    not_empty_veh = np.any(solution.Y,axis=3)
    candidates = np.transpose(np.where(np.sum(not_empty_veh, axis = 2) > 1))
    if len(candidates) > 0:
        for i in range(rho): 
            [t,n] = random.choice(candidates)
            candid_veh = np.where(not_empty_veh[t,n,:])[0]
            number = len(candid_veh)
            k1, k2 = candid_veh[np.random.choice(number,2,replace= False)]
            m1, m2 = random.choice(np.nonzero(solution.Y[t,n,k1,:])[0]), random.choice(np.nonzero(solution.Y[t,n,k2,:])[0])
            if m1 != m2 :
                solution.Y[t,n,[k1, k2],m1] = [0, 1]
                solution.Y[t,n,[k1, k2],m2] = [1, 0]
            solution.r[t][n][k1], solution.r[t][n][k2] = tsp_tour(np.nonzero(solution.Y[t,n,k1,:])[0] + solution.N, n, solution.problem.D), tsp_tour(np.nonzero(solution.Y[t,n,k2,:])[0] + solution.N, n, solution.problem.D)
    #else:
    #    print('Applied swap_rho_cust_intra_routes, but there were no viable candidates (=Warehouses with two departing vehicles in one time step')
        
def swap_rho_cust_intra_plants(solution, rho):
    max_iter = 100
    iterations = 0
    changed = 0
    candidates_time = np.where(np.sum(np.any(solution.Y, axis = (2,3)), axis = 1) > 1)[0]
    if len(candidates_time) > 0:
        while iterations < max_iter and changed < rho :
            t = random.choice(candidates_time)
            candidates_warehouses = np.nonzero(np.any(solution.Y[t,:,:,:], axis = (1,2)))[0]
            temp1, temp2 = np.random.choice(len(candidates_warehouses), 2, replace = False)
            n1, n2 = candidates_warehouses[temp1], candidates_warehouses[temp2]
            candidates_1 = np.transpose(np.nonzero(solution.Y[t,n1,:,:]))
            candidates_2 = np.transpose(np.nonzero(solution.Y[t,n2,:,:]))
            [k1,m1], [k2,m2] = random.choice(candidates_1), random.choice(candidates_2)
            if solution.Cl[n1,m2] == 1 and solution.Cl[n2,m1] == 1:
                solution.Y[t,[n1, n2],[k1, k2],m1] = [0, 1]
                solution.Y[t,[n1, n2],[k1, k2],m2] = [1, 0]
                solution.r[t][n1][k1], solution.r[t][n2][k2] = tsp_tour(np.nonzero(solution.Y[t,n1,k1,:])[0] + solution.N, n1, solution.problem.D), tsp_tour(np.nonzero(solution.Y[t,n2,k2,:])[0] + solution.N, n2, solution.problem.D)
                changed += 1
            iterations += 1
        


    

operators = [
        ('rand remove rho',rand_remove_rho),
        ('remove worst rho',remove_worst_rho),
        ('shaw removal route based',shaw_removal_route_based),
        ('shaw removal greedy',shaw_removal_greedy),
        ('avoid consecutive visits',avoid_consecutive_visits),
        ('empty one period',empty_one_period),
        ('empty one vehicle',empty_one_vehicle),
        ('empty one plant',empty_one_plant),
        ('further customer', furthest_customer),
        ('rand insert rho',rand_insert_rho),
        ('assign to nearest plant',assign_to_nearest_plant),
        ('insert best rho',insert_best_rho),
        ('shaw insertion', shaw_insertion),
        ('swap rho cust intra routes',swap_rho_cust_intra_routes),
        ('swap rho cust intra plants',swap_rho_cust_intra_plants)
]


#remove_worst_rho
#insert_best_rho
#assign_to_nearest_plant
