#test code
import sys
sys.path.append('../')
from ISI import excel_to_pb, Matheuristic, Meta_param, Problem, cluster_fusing
from os.path import expanduser as ospath


t_load = 0.5
time_step = 2
cooling = 0.2
T = 4
H = 2
t_virt = 1

param = Meta_param(seed=1)
param.tau_start = 3.
param.tau_end = 1.
param.cooling = cooling




file = ospath("~/Desktop/GitHub/IRP-framework/Data/Burundi final v2.xlsx")   # leo
file = ospath("~/Desktop/GitHub/IRP-framework/Data/Burundi final v2.xlsx")   # chris

schools, warehouses, Q1, V_number, makes = excel_to_pb(file,nbr_tours=1)

problem_global = Problem(Schools = schools, Warehouses = warehouses,
                                T = T, Q1 = Q1, Q2 = 20, v = 40,
                                t_load = t_load, c_per_km = 1, Tmax = 10, V_number = V_number,
                                central = None, makes = makes, H=H, t_virt = t_virt)

problem_global = problem_global.time_defuse(time_step)
problems = problem_global.clustering()
solutions = []
for counter, pr in enumerate(problems):
    heuristic = Matheuristic(pr,param=param)
    heuristic.algo2(plot_final=True, file = "solution/cluster %i.html" % (counter+1) )
    solutions.append(heuristic.solution_best)
    print('Cluster {} of {} computed!'.format(counter + 1, len(problems)))

solution = cluster_fusing(solutions,problem_global)
solution.file = "solution/global.html"
print(solution)
            