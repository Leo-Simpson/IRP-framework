#test code
import sys
sys.path.append('../')
from ISI import excel_to_pb, Matheuristic, Meta_param, Problem, cluster_fusing

from os.path import expanduser as ospath

file = ospath("~/Desktop/GitHub/IRP-framework/Data/Burundi final v2.xlsx")

schools, warehouses, Q1, V_number, makes = excel_to_pb(file,nbr_tours=1)

problem_global = Problem(Schools = schools, Warehouses = warehouses,
                                T = 2, Q1 = Q1, Q2 = 20, v = 40,
                                t_load = 0.5, c_per_km = 1, Tmax = 10, V_number = V_number,
                                central = None, makes = makes)

problem_global = problem_global.time_defuse(2)
problems = problem_global.clustering()
param = Meta_param(seed=1)
param.tau_start = 3.
param.tau_end = 1.
param.cooling = 0.2
solutions = []
for counter, pr in enumerate(problems):
    heuristic = Matheuristic(pr,param=param)
    heuristic.algo2(plot_final=True, file = "solution/cluster %i.html" % (counter+1) )
    solutions.append(heuristic.solution_best)
    print('Cluster {} of {} computed!'.format(counter + 1, len(problems)))

solution = cluster_fusing(solutions,problem_global)
solution.file = "solution/global.html"
print(solution)
            