#test code
import sys
sys.path.append('../')
from ISI import random_problem, Matheuristic


T, N, K,M = 5,2,4,15
H = 5
problem = random_problem(T,N,M,K, H = H,seed=1)
heuristic = Matheuristic(problem)
heuristic.param.rho_percent = 0.3
heuristic.param.cooling = 0.5

heuristic.algo2(info = True, plot = False)

