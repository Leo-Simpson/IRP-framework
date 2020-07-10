#test code
import sys
sys.path.append('../')
from ISI import random_problem, Matheuristic, Meta_param


T, N, K,M = 5,2,None,20
H = 5
problem = random_problem(T,N,M,K, H = H,seed=1)
# heuristic = Matheuristic(problem)
# heuristic.param.rho_percent = 0.3
# heuristic.param.cooling = 0.9

# heuristic.algo2(info = False, plot = False)

problems = problem.clustering()
param = Meta_param(seed=1)
param.tau_start = 3.
param.tau_end = 1.
for counter, pr in enumerate(problems):
    heuristic = Matheuristic(pr,param=param)
    heuristic.algo2(info = True)
    print('Cluster {} of {} computed!'.format(counter + 1, len(problems)))

