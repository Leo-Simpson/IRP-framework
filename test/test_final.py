#test code
import sys
sys.path.append('../')
from ISI import random_problem, Matheuristic, Meta_param, cluster_fusing


T, N, K,M = 5,2,None,15
H = 5
problem = random_problem(T=T,N=N,M=M,K=K, H = H,seed=1)
heuristic = Matheuristic(problem)
heuristic.param.rho_percent = 0.3
heuristic.param.cooling = 0.9

heuristic.algo2(info = True, plot = False)

# problems = problem.clustering()
# param = Meta_param(seed=1)
# param.tau_start = 3.
# param.tau_end = 1.
# param.cooling = 0.8
# solutions = []
# for counter, pr in enumerate(problems):
#     heuristic = Matheuristic(pr,param=param)
#     heuristic.algo2(plot_final=True, file = "solution/cluster %i.html" % (counter+1) )
#     solutions.append(heuristic.solution_best)
#     print('Cluster {} of {} computed!'.format(counter + 1, len(problems)))

# solution = cluster_fusing(solutions,problem)

