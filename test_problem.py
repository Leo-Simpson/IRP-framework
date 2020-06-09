#test code

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from ISI import Problem, Solution


def test_problem(M, N, T, K):
    
    Schools = []
    Warehouses = []

    np.random.seed(10)

    for i in range(M):
        Schools.append((np.random.randint(low = 10, high = 100)*10, np.random.randint(low = 1, high = 10)*10, 'School {}'.format(i+1), np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])))


    for i in range(N):
        Warehouses.append((np.random.randint(low = 1, high = 100)*1000, np.random.randint(low = 1, high = 10)*100, 'Warehouse {}'.format(i+1), np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])))


    locations = [w[3] for w in Warehouses] + [s[3] for s in Schools]
    names = [w[2] for w in Warehouses] + [s[2] for s in Schools]

    distance_mat = np.round(distance_matrix(locations,locations))

    D = pd.DataFrame(distance_mat, columns = names, index=names)


    problem = Problem(D = D, Schools = Schools, Warehouses = Warehouses,T = T,K = K, Q1 = 1000, Q2 = 5000, v = 40, t_load = 0.5, c_per_km = 1)
    return problem
