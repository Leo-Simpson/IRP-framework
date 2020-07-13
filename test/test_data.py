#test code
import sys
sys.path.append('../')
from ISI import excel_to_pb, Matheuristic, Meta_param, Problem, cluster_fusing
from os.path import expanduser as ospath


t_load = 0.5
time_step = 0.25
cooling = 0.8
T = 1
H = None
t_virt = 1

param = Meta_param(seed=1)
param.tau_start = 3.
param.tau_end = 1.
param.cooling = cooling


file = "../Data/Burundi final v2.xlsx"

schools, warehouses, Q1, V_number, makes = excel_to_pb(file,nbr_tours=1)

problem_global = Problem(Schools = schools, Warehouses = warehouses,
                                T = T, Q1 = Q1, Q2 = 20, v = 40,
                                t_load = t_load, c_per_km = 1, Tmax = 10, V_number = V_number,
                                central = None, makes = makes, H=H, t_virt = t_virt)

problem_global.final_solver(param,time_step=time_step)
            