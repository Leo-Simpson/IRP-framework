#test code
import sys

sys.path.append('../')
from ISI import test_problem, Solution


T, N, K,M = 5,2,4,6
problem = test_problem(T,N,K,M)

solution = Solution(problem)

solution.ISI(G = N)

print(solution)




