#test code

from test_problem import test_problem
from ISI import Problem, Solution
from visu import visu 

import plotly.graph_objects as go 
from plotly import offline



T, N, K,M = 10,2,4,6
problem = test_problem(T,N,K,M)
I_s = [s["initial"] for s in problem.Schools]
I_w = [w["initial"] for w in problem.Warehouses]
cost = 10
total_cost = 30

title = TITLE + "        Cost = %s          Total Cost = " %str(cost) + str(total_cost)
title = title + "   Truck 1 capacity : "+ str(problem.Q1) + "   Truck 2 capacity : "+ str(problem.Q2)



visual = visu(problem,title, I_s,I_w)
fig = go.Figure(visual)
offline.plot(fig, filename= "visu.html")

