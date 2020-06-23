#test code
import numpy as np
from test_problem import test_problem
from ISI import Problem, Solution
from visu import visu 

import plotly.graph_objects as go 
from plotly import offline



T, N, K,M = 10,2,4,6
problem = test_problem(T,N,K,M)
I_s = [np.array([s["initial"] for s in problem.Schools])]
I_w = [np.array([w["initial"] for w in problem.Warehouses])]


costs = [10]*T

for t in range(T-1):
    I_s.append(I_s[t]-1)
    I_w.append(I_w[t]-1)



visual = visu(problem,"WFP Inventory problem", I_s,I_w, costs)

fig = go.Figure(visual)
offline.plot(fig, filename= "visu.html")

