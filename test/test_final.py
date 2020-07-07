#test code
import sys
sys.path.append('../')
from ISI import random_problem, Matheuristic


T, N, K,M = 4,2,4,15
H = 4
problem = random_problem(T,N,K,M, H = H,seed=1)
heuristic = Matheuristic(problem)
heuristic.param.rho_percent = 0.3
heuristic.param.cooling = 0.5

heuristic.algo2(info = False, plot = False)

