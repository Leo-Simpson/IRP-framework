import numpy as np
import numpy.random as rd
from copy import deepcopy
#edit Chris 07.06.20:
from chris_version_solve_tsp import tsp_tour
#end of edit Chris 07.06.20:

class Problem :
    #this is the class that contains the data of the problem
    def __init__(self,D,Warehouses,Schools,T,K, Q):
        self.D = D # distance matrix. Could be a pandas data frame with the names of Warehouses/Schools as index of rows and colomns 
                    # to get the distance between a warehouse and a school for example : D.loc[warehouse_name, school_name]
        self.Warehouses = Warehouses # list of tuple (capacity,fixed_cost, name)
        self.Schools = Schools  # list of tuple (capactiy, consumption, name)
        self.T = T # time horizon
        self.K = K # number of vehicles
        self.Q = Q # capacity of the trucks 


class Solution : 
    def __init__(self,problem):
        M,N,K,T = len(problem.Schools), len(problem.Warehouses),problem.K, problem.T
        self.M, self.N, self.K, self.T = M,N,K,T

        self.name_schools    = [s[2] for s in problem.Schools ]
        self.name_warehouses = [w[2] for w in problem.Warehouses ]
        
        self.problem = problem

        self.I_s = np.zeros((T,M))  # invetory of the schools
        self.I_w = np.zeros((T,N))  # inventory        
        self.Y = np.zeros((T,K,M,N), dtype = bool) # variable  equal 1 if vehicle k delivers school m from warehouse n at time t
        

        # other variable to add probably


    def compute_a_and_b(self):
        # to do 
        self.a = None # routing cost reduction if customer l is removed from the tour of vehicle k at time t ==> array TxKxM
        self.b = None # routing cost addition if customer l is added to the tour of vehicle k at time t ==> array TxKxM

    def compute_r(self):
        # here are the TSP to be computed
        self.r = [[[ ]*K]*T] # for each time t, for each vehicle k, a list of ordered integer of [0, M+N] corresponding to a tour
        for t in range(self.T):
            for k in range(self.K):
                w,s = np.nonzero(self.Y[t,k,:,:])
                tour = [w[0]]+ list(s+N) #  the tour starts at the warehouse then add the school in the wrong order
                

                # then do something for solving the associated TSP. 
                
                #edit Chris 07.06.20:
                tour = tsp_tour(tour, self)
                #end of edit Chris 07.06.20:


                self.r[t][k] = tour

    def compute_cost(self):
        self.cost = 0. # to do 

    def ISI(self, G = 1):
        # change the solution itself to the ISI solution
        self.compute_a_and_b()


        # TO DO HERE !!!
        self.q = None
        self.delta = None
        self.omega = None
        # To do here, with a solver ... 


        self.compute_r()
        self.compute_cost()




class Meta_param : 
    def __init__(self):
        self.Delta = 20
        self.epsilon_bound = (0.05,0.15)
        self.tau_start = 8000
        self.tau_end = 0.01
        self.cooling = 0.989
        self.reaction_factor = 0.8
        self.sigmas = (10,5,2)
        self.ksi = rd.uniform(low=0.1,high=0.2)
        self.seed = None


class Matheuristic : 
    def __init__(self, initial_solution):
        self.operators = [
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.operator1, 'name': 'operator1' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.operator2, 'name': 'operator2' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.operator3, 'name': 'operator3' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.operator4, 'name': 'operator4'}
        ]

        self.solution = initial_solution
        self.solution_best = deepcopy(initial_solution)
        self.solution_prime = deepcopy(initial_solution)


    def final_algo(self, param):
        # here one can do the final matheuristic described in the paper
        M,N,K,T = self.solution.M, self.solution.N, self.solution.K, self.solution.T
        
        # initialization (step 2 and 3 of the pseudo code)
        self.solution.ISI(G = N+M)
        self.solution_best = deepcopy(self.solution)

        # line 4 of pseudocode
        epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1], seed = param.seed  )

        tau = param.tau_start
        iterations = 0
        while tau > param.tau_end and iterations < 1000 : 
            
            # line 6 of pseudocode
            i = Matheuristic.choose_operator(self.operators)
            operator = self.operators[i]['function']
            self.solution_prime = deepcopy(self.solution)
            operator(self.solution_prime)
            G = N+M
            self.solution_prime.ISI(G=G)

            if self.solution_prime.cost < self.solution.cost : # line 7
                self.solution = deepcopy(self.solution_prime) # line 8
                G = max(G-1,1)                                  # line 9

                keep_going = True
                while keep_going :                               #line 10
                    self.solution_prime.ISI(G=G)  
                    if self.solution_prime.cost < (1+epsilon)*self.solution.cost: 
                        if self.solution_prime.cost < self.solution.cost :              # line 11
                            self.solution = deepcopy(self.solution_prime)               # line 12
                            G = max(G-1,1)                                              # line 13
                        else : G = max(int(param.ksi*(N+M)),1)                          # line 14-15
                                            

                    elif self.solution.cost < self.solution_best.cost :   # line 17
                        self.solution_best = deepcopy(self.solution)    # line 18
                        self.operators[i]['score'] += param.sigmas[0]   # line 19
                        G = max(G-1,1)                                  # line 20
                    else :                                              # line 21
                        self.operators[i]['score'] += param.sigmas[1]   # line 22

                        # little deviation from the pseudocode... we think it is s and not s''
                        if self.solution.cost < (1+epsilon)*self.solution_best :  # line 23
                            G = max(int(param.ksi*(N+M)),1)                       # line 24
                        else : keep_going = False

            elif self.solution_prime.cost < self.solution.cost - np.log(rd.random())*tau: # line 27
                self.solution = deepcopy(self.solution_prime)                             # line 28
                self.operators[i]['score'] += param.sigmas[2]                             # line 29
            
            if iterations % param.delta == 0 :
                epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1], seed = param.seed  )
                self.update_weights(param.reaction_factor)
                self.solution = deepcopy(self.solution_best)
            iterations += 1

        

                





    def operator1(solution):
        pass

    def operator2(solution):
        pass

    def operator3(solution):
        pass

    def operator4(solution):
        pass


    def choose_operator(operators):
        i = 0 # to do
        
        return i

    def update_weights(self, r):        # r is the reaction factor
        for op in self.operators : 
            if (op['number_used']>0): op['weight'] = (1-r)*op['weight'] + r* op['score']/op['number_used']
            op['score']  = 0





'''
TO DO LIST : 

- Do a fake ISI function to test our Matheuristic
- Find how to do the TSPs, and compute_r
- Write the ISI function
- Write the operators
- choose_operator
- compute_a_and_b
'''




