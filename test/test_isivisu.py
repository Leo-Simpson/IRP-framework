#test code
import sys
sys.path.append('../')
from ISI import random_problem, Solution


T, N, K,M = 6,2,4,6
problem = random_problem(T,N,K,M, seed=10)
solver = "CBC"
G = 6

solution = Solution(problem)
print("G = ", G)
solution.ISI(G = G, solver = solver)

G = G-1

print("G = ", G)
solution.ISI(G = G, solver = solver)
G = G-1

print("G = ", G)
solution.ISI(G = G, solver = solver)
G = G-1

print("G = ", G)
solution.ISI(G = G, solver = solver, plot = True)




