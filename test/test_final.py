#test code
import sys
sys.path.append('../')
from ISI import random_problem, Meta_param, Matheuristic


T, N, K,M = 4,2,4,6
problem = random_problem(T,N,K,M, seed=10)
param = Meta_param()
heuristic = Matheuristic(problem)

heuristic.final_algo(param, MAXiter = 10, solver = "CBC")



