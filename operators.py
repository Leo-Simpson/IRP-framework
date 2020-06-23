import numpy as np

def rand_remove_rho(solution, rho):
    for i in range(np.min([rho,len(np.nonzero(solution.Y)[0])])):  
        t,n,k,m = random.choice(np.transpose(np.nonzero(solution.Y)))
        solution.Y[t,n,k,m] = 0
        tour = np.array(solution.r[t][n][k])
        solution.r[t][n][k] = np.ndarray.tolist(tour[tour != m + solution.N])
        

def remove_worst_rho(solution, rho):
    for i in range(np.min([rho,len(np.nonzero(solution.Y)[0])])):   
        Y_flat = solution.Y.reshape(-1)
        a_flat = solution.a.reshape(-1)
        argmax = np.argmax(a_flat[Y_flat == 1])
        choice = np.where(Y_flat == 1)[0][argmax]
        Y_flat[choice] = 0
        a_flat[choice] = 0
        t, rest = np.divmod(choice, solution.N*solution.K*solution.M)
        n, rest = np.divmod(rest, solution.K*solution.M)
        k,m = np.divmod(rest, solution.M)
        tour = np.array(solution.r[t][n][k])
        solution.r[t][n][k] = np.ndarray.tolist(tour[tour != m + solution.N])
        solution.compute_school_remove_costs(t,n,k)

def shaw_removal_route_based(solution, rho):
    t,n,k,m = random.choice(np.transpose(np.nonzero(solution.Y)))
    route = np.array(solution.r[t][n][k])
    if len(route) > 2:
        schools = route[np.where(route != m + solution.N)[0]]
        dist_from_m = solution.problem.D.values[np.ix_([m + solution.N],route)][0]
        min_dist_from_m = np.min(solution.problem.D.values[np.ix_([m + solution.N],schools)][0])
        to_remove = route[np.where(dist_from_m <= 2*min_dist_from_m)[0]] - solution.N 
        solution.Y[t,n,k,to_remove] = 0
        solution.r[t][n][k] = tsp_tour(np.setdiff1d(route, to_remove + solution.N), n, solution.problem.D.values)
    else:
        solution.Y[t,n,k,:] = 0
        solution.r[t][n][k] = []


def shaw_removal_greedy(solution, rho):
    t,n,k,m = random.choice(np.transpose(np.nonzero(solution.Y)))
    route = np.array(solution.r[t][n][k])
    if len(route) > 2:
        schools = route[np.where(route != m + solution.N)[0]]
        dist_from_m = solution.problem.D.values[np.ix_([m + solution.N],route)][0]
        dist = solution.problem.D.values[np.ix_(route,route)]
        min_dist = np.min(dist[dist>0])
        to_remove = route[np.where(dist_from_m <= 2*min_dist)[0]] - solution.N 
        solution.Y[t,n,k,to_remove] = 0
        solution.r[t][n][k] = tsp_tour(np.setdiff1d(route, to_remove + solution.N), n, solution.problem.D.values)
    else:
        solution.Y[t,n,k,:] = 0
        solution.r[t][n][k] = []

def avoid_consecutive_visits(solution, rho):
    for t in range(solution.T-1):
        time_schools = np.sum(solution.Y[[t, t+1],:,:,:], axis = (1,2))
        index = np.where(time_schools[1,:] + time_schools[0,:] > 1)
        solution.Y[t+1,:,:,index] = 0

def empty_one_period(solution, rho):
    period = np.random.randint(solution.T)
    solution.Y[period,:,:,:] = 0

def empty_one_vehicle(solution, rho):
    warehouse = np.random.randint(solution.N)
    vehicle = np.random.randint(solution.K)
    solution.Y[:,warehouse,vehicle,:] = 0

def empty_one_plant(solution, rho):
    warehouse = np.random.randint(solution.N)
    solution.Y[:,warehouse,:,:] = 0
    
def furthest_customer(solution, rho):
    candidates = np.sum(solution.Y, axis = 3)
    for i in range(np.min([rho,np.sum(candidates)])):
        t,n,k = random.choice(np.transpose(np.nonzero(candidates)))
        route = solution.r[t][n][k]
        furthest_cust_index = np.argmax(solution.problem.D.values[np.ix_([n],route)][0])
        solution.Y[t,n,k,route[furthest_cust_index] - solution.N] = 0
        solution.r[t][n][k] = np.delete(route, furthest_cust_index)
        candidates[t,n,k] -= 1
    
def rand_insert_rho(solution, rho):    
    candidates = 1-np.sum(solution.Y, axis = (1,2))
    for i in range(np.min([rho,np.sum(candidates)])):
        t, m = random.choice(np.transpose(np.nonzero(candidates)))
        candidates[t,m] -= 1
        n, k = random.choice(np.nonzero(solution.Cl[:,m])[0]), np.random.randint(solution.K)
        solution.Y[t,n,k,m] = 1
        solution.r[t][n][k],_ = solution.cheapest_school_insert(t,n,k,m)
    
def assign_to_nearest_plant(solution, rho):
    candidates = 1-np.sum(solution.Y, axis = (1,2))
    for i in range(np.min([rho,np.sum(candidates)])):
        t, m = random.choice(np.transpose(np.nonzero(candidates)))
        allowed_plants = [i for i in range(solution.N) if solution.Cl[i,m] == 1]
        nearest_plant = allowed_plants[np.argmin(solution.problem.D.values[np.ix_([m + solution.N],allowed_plants)][0])]
        candidates[t,m] = 0
        route, cost = solution.cheapest_school_insert(t,nearest_plant,0,m)
        index = 0
        for k in range(1, solution.K):
            route_temp, cost_temp = solution.cheapest_school_insert(t,nearest_plant,k,m)
            if cost_temp < cost:
                route = route_temp
                index = k
        solution.Y[t,nearest_plant,k,m] = 1
        solution.r[t][nearest_plant][k] = route
    
def insert_best_rho(solution, rho):
    candidates = 1-np.sum(solution.Y, axis = (1,2))
    for i in range(np.min([rho,np.sum(candidates)])): 
        b_flat = solution.b.reshape(-1)
        Y_flat = solution.Y.reshape(-1)
        choice = np.where(b_flat>0)[0][np.argmin(b_flat[b_flat>0])]
        Y_flat[choice] = 1
        t, rest = np.divmod(choice, solution.N*solution.K*solution.M)
        n, rest = np.divmod(rest, solution.K*solution.M)
        k, m = np.divmod(rest, solution.M)
        solution.b[t,:,:,m] = 0
        solution.r[t][n][k],_ = solution.cheapest_school_insert(t,n,k,m)
        #tour_school = np.nonzero(solution.Y[t,n,k,:])[0] + solution.N 
        #solution.r[t][n][k] = tsp_tour(tour_school, n, solution.problem.D.values)
        solution.compute_school_insert_costs(t,n,k)
    
def shaw_insertion(solution, rho): 
    dist = solution.problem.D.values
    period = np.random.randint(solution.T)
    not_served = np.where(np.sum(solution.Y[period,:,:,:], axis = (0,1)) == 0)[0]
    (index, choice) = random.choice(list(enumerate(not_served)))
    dist_to_all = dist[np.ix_([choice + solution.N],[m  + solution.N for m in range(solution.M) if m != choice])][0]
    rest_not_served = np.delete(not_served, index)
    dist_to_not_served = dist[np.ix_([choice + solution.N],rest_not_served + solution.N)][0]
    close = rest_not_served[dist_to_not_served <= 2*np.min(dist_to_all)]
    closest_warehouse = np.argmin(dist[np.ix_([choice + solution.N],[i for i in range(solution.N)])][0])
    close_reachable = [m for m in close[solution.Cl[closest_warehouse, close] == 1]]
    route = tsp_tour(close_reachable + solution.r[period][closest_warehouse][0],closest_warehouse,dist)
    costs = solution.compute_route_dist(route, closest_warehouse)
    index = 0
    for k in range(1, solution.K):
        route_temp = tsp_tour(close_reachable + solution.r[period][closest_warehouse][k],closest_warehouse,dist)
        costs_temp = solution.compute_route_dist(route_temp, closest_warehouse)
        if costs_temp < costs:
            route = route_temp
            index = k
    solution.Y[period,closest_warehouse,index, close_reachable] = 1
    solution.r[period][closest_warehouse][index] = route
    
def swap_rho_cust_intra_routes(solution, rho):
    dist = solution.problem.D.values
    sum_over_schools = np.sum(solution.Y, axis = 3)
    candidates = np.transpose(np.where(np.count_nonzero(sum_over_schools, axis = 2) > 1))
    for i in range(rho): 
        [t,n] = random.choice(candidates)
        candid_veh = np.where(sum_over_schools[t,n,:] > 0)[0]
        number = len(candid_veh)
        k1, k2 = candid_veh[np.random.choice(number,2,replace= False)]
        m1, m2 = random.choice(np.nonzero(solution.Y[t,n,k1,:])[0]), random.choice(np.nonzero(solution.Y[t,n,k2,:])[0])
        solution.Y[t,n,[k1, k2],m1] = [0, 1]
        solution.Y[t,n,[k1, k2],m2] = [1, 0]
        solution.r[t][n][k1], solution.r[t][n][k2] = tsp_tour(np.nonzero(solution.Y[t,n,k1,:])[0] + solution.N, n, dist), tsp_tour(np.nonzero(solution.Y[t,n,k2,:])[0] + solution.N, n, dist)
        
def swap_rho_cust_intra_plants(solution, rho):
    max_iter = 100
    iterations = 0
    changed = 0
    dist = solution.problem.D.values
    candidates_warehouses = np.transpose(np.nonzero(np.sum(solution.Y, axis = (2,3))))
    candidates_time = np.where(np.count_nonzero(np.sum(solution.Y, axis = (2,3)), axis = 1) > 1)[0]
    while iterations < max_iter and changed < rho :
        t = random.choice(candidates_time)
        candidates_warehouses = np.nonzero(np.sum(solution.Y[t,:,:,:], axis = (1,2)))[0]
        temp1, temp2 = np.random.choice(len(candidates_warehouses), 2, replace = False)
        n1, n2 = candidates_warehouses[temp1], candidates_warehouses[temp2]
        candidates_1 = np.transpose(np.nonzero(solution.Y[t,n1,:,:]))
        candidates_2 = np.transpose(np.nonzero(solution.Y[t,n2,:,:]))
        [k1,m1], [k2,m2] = random.choice(candidates_1), random.choice(candidates_2)
        if solution.Cl[n1,m2] == 1 and solution.Cl[n2,m1] == 1:
            solution.Y[t,[n1, n2],[k1, k2],m1] = [0, 1]
            solution.Y[t,[n1, n2],[k1, k2],m2] = [1, 0]
            solution.r[t][n1][k1], solution.r[t][n2][k2] = tsp_tour(np.nonzero(solution.Y[t,n1,k1,:])[0] + solution.N, n1, dist), tsp_tour(np.nonzero(solution.Y[t,n2,k2,:])[0] + solution.N, n2, dist)
            changed += 1
        iterations += 1


    

operators = [
        ('rand_remove_rho',rand_remove_rho),
        ('remove_worst_rho',remove_worst_rho),
        ('shaw_removal_route_based',shaw_removal_route_based),
        ('shaw_removal_greedy',shaw_removal_greedy),
        ('avoid_consecutive_visits',avoid_consecutive_visits),
        ('empty_one_period',empty_one_period),
        ('empty_one_vehicle',empty_one_vehicle),
        ('empty_one_plant',empty_one_plant),
        ('rand_insert_rho',rand_insert_rho),
        ('assign_to_nearest_plant',assign_to_nearest_plant),
        ('insert_best_rho',insert_best_rho),
        ('swap_rho_cust_intra_routes',swap_rho_cust_intra_routes),
        ('swap_rho_cust_intra_plants',swap_rho_cust_intra_plants)
]