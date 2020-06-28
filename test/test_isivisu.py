#test code
import sys
sys.path.append('../')
from ISI import random_problem, Solution


T, N, K,M = 6,2,4,6
problem = random_problem(T,N,K,M, seed=10)
solver = "CBC"
G = 6
print( "Solver : ", solver)

solution = Solution(problem)
print("G = ", G)
solution.ISI(G = G, solver = solver)
solution.verify_feasibility()
print(solution)
G = G-1

print("G = ", G)
solution.ISI(G = G, solver = solver )
solution.verify_feasibility()
print(solution)
G = G-1

print("G = ", G)
solution.ISI(G = G, solver = solver)
solution.verify_feasibility()
print(solution)
G = G-1

print("G = ", G)
solution.ISI(G = G, solver = solver)
solution.verify_feasibility()
print(solution)
G = G-1



