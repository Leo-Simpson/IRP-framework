#test code
import sys
sys.path.append('../')
from ISI import random_problem, Meta_param, Matheuristic


T, N, K,M = 5,2,4,8
problem = random_problem(T,N,K,M, seed=10)
param = Meta_param(seed = 1)
heuristic = Matheuristic(problem)

heuristic.algo2(param)



