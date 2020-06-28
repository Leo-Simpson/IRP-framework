#test code
import sys
sys.path.append('../')
from ISI import random_problem, Solution


T, N, K,M = 5,2,4,6
problem = random_problem(T,N,K,M, seed=42)

solution = Solution(problem)


solution.ISI(G = 6, accuracy = 0.1)
print(solution)



