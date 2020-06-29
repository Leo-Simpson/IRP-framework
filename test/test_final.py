#test code
import sys
sys.path.append('../')
from ISI import random_problem, Matheuristic


T, N, K,M = 5,2,4,10
problem = random_problem(T,N,K,M, seed=3)
heuristic = Matheuristic(problem)

heuristic.algo2()



