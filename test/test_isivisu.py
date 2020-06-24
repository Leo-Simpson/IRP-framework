#test code
import sys
sys.path.append('../')
from ISI import random_problem, Solution


T, N, K,M = 10,2,4,6
problem = random_problem(T,N,K,M, seed=6)

solution = Solution(problem)

solution.ISI(G = 6)

print(solution)

solution.ISI(G = 3)

print(solution)

solution.ISI(G=2)

print(solution)




