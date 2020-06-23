#test code
import sys

sys.path.append('../')
from ISI import random_problem, Solution


T, N, K,M = 10,2,4,6
problem = random_problem(T,N,K,M, seed=10)

solution = Solution(problem)

solution.ISI(G = N)

print(solution)




