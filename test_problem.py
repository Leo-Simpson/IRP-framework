#test code

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from ISI import Problem, Solution


def test_problem(T,N,K,M):
    
    Schools = []
    Warehouses = []

    np.random.seed(10)

    for i in range(M):
        Schools.append({'capacity': np.random.randint(low = 10, high = 100)*10, 'lower': np.random.randint(low = 1, high = 5)*100, 'consumption': np.random.randint(low = 1, high = 10)*10 ,'storage_cost': np.random.randint(low = 1, high = 5)*10 , 'initial':  np.random.randint(low = 1, high = 10)*100,  'name' : 'School {}'.format(i+1), 'location': np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])})


    for i in range(N):
        Warehouses.append({'capacity': np.random.randint(low = 1, high = 100)*1000, 'lower': np.random.randint(low = 10, high = 20)*1000 , 'dist_central': np.random.randint(low = 1, high = 100) , 'fixed_cost': np.random.randint(low = 1, high = 10)*100 , 'initial': np.random.randint(low = 1, high = 20)*1000,  'name' : 'Warehouse {}'.format(i+1), 'location': np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])})


    locations = [w['location'] for w in Warehouses] + [s['location'] for s in Schools]
    names = [w['name'] for w in Warehouses] + [s['name'] for s in Schools]

    distance_mat = np.round(distance_matrix(locations,locations))

    D = pd.DataFrame(distance_mat, columns = names, index=names)


    problem = Problem(D = D, Schools = Schools, Warehouses = Warehouses,T = T,K = K, Q1 = 1000, Q2 = 5000, v = 40, t_load = 0.5, c_per_km = 1, Tmax = 6)
    return problem
