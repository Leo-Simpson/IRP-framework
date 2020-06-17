import numpy as np
from ISI import Problem, Solution, Matheuristic
from test_problem import test_problem


T,N,K,M = 5, 2, 3, 5
problem = test_problem(T,N,K,M)

solution = Solution(problem)

solution.ISI(G = N+M)






