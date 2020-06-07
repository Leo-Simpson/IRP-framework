#test code

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from ISI import Problem, Solution


Schools = []
Warehouses = []

np.random.seed(10)

for i in range(12):
    Schools.append((np.random.randint(low = 10, high = 100)*10, np.random.randint(low = 1, high = 10)*10, 'School {}'.format(i+1), np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])))
    

for i in range(4):
    Warehouses.append((np.random.randint(low = 1, high = 100)*1000, np.random.randint(low = 1, high = 10)*100, 'Warehouse {}'.format(i+1), np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])))


locations = [w[3] for w in Warehouses] + [s[3] for s in Schools]
names = [w[2] for w in Warehouses] + [s[2] for s in Schools]

distance_matrix = np.round(distance_matrix(locations,locations),1)

D = pd.DataFrame(distance_matrix, columns = names, index=names)


problem = Problem(D = D, Schools = Schools, Warehouses = Warehouses,T = 6,K = 3, Q = 1000, v = 40, t_load = 0.5, c_per_km = 1)
solution = Solution(problem)
