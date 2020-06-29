#test code
import sys
sys.path.append('../')
from ISI import random_problem, Meta_param, Matheuristic


T, N, K,M = 4,2,4,6
problem = random_problem(T,N,K,M, seed=10)
problem.Tmax = 50
param = Meta_param()
heuristic = Matheuristic(problem)

heuristic.algo2(param, MAXiter = 10, solver = "CBC")



