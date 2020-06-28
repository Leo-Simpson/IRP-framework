#test code
import sys
sys.path.append('../')
from ISI import random_problem, Solution


T, N, K,M = 5,2,4,6
problem = random_problem(T,N,K,M, seed=25)

solution = Solution(problem)
solution.ISI(G = 6, solver = "CBC")
solution.verify_feasibility()
print(solution)

solution.ISI(G = 5, solver = "CBC")
solution.verify_feasibility()
print(solution)

solution.ISI(G = 4, solver = "CBC")
solution.verify_feasibility()
print(solution)

solution.ISI(G = 3, solver = "CBC")
solution.verify_feasibility()
print(solution)



