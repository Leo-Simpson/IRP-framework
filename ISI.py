import numpy as np
from copy import deepcopy

class Problem :
    #this is the class that contains the data of the problem
    def __init__(self,D,Warehouses,Schools,T,K, Q):
        self.D = D # distance matrix. Could be a pandas data frame with the names of Warehouses/Schools as index of rows and colomns 
                    # to get the distance between a warehouse and a school for example : D.loc[warehouse_name, school_name]
        self.Warehouses = Warehouses # list of tuple (capacity,fixed_cost, name)
        self.Schools = Schools  # list of tuple (capactiy, consumption, name)
        self.T = T # time horizon
        self.K = K # number of vehicules
        self.Q = Q # capacity of the trucks 


class Solution : 
    def __init__(self,problem):
        M,N,K,T = len(problem.Schools), len(problem.Warehouses),problem.K, problem.T
        self.M, self.N, self.K, self.T = M,N,K,T

        self.name_schools    = [s[2] for s in problem.Schools ]
        self.name_warehouses = [w[2] for w in problem.Warehouses ]
        
        self.problem = problem

        self.operators = [
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator1 },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator2 },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator3 },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator4 }
        ]



        self.I_s = np.zeros((T,M))  # invetory of the schools
        self.I_w = np.zeros((T,N))  # inventory        
        self.Y = np.zeros((T,K,M,N), dtype = bool) # variable  equal 1 if vehicle k delivers school m from warehouse n at time t
        

        # other variable to add probably


    def compute_a_and_b(self):
        # to do 
        self.a = None
        self.b = None
        pass

    def compute_r(self):
        # here are the TSP to be computed
        self.r = [[[ ]*K]*T] # for each time t, for each vehicle k, a list of ordered integer of [0, M+N] corresponding to a tour
        for t in range(self.T):
            for k in range(self.K):
                w,s = np.nonzero(self.Y[t,k,:,:])
                tour = [w[0]]+ list(s+N) #  the tour starts at the warehouse then add the school in the wrong order
                

                # then do something for solving the associated TSP. 


                self.r[t][k] = tour

    def ISI(self, G = 1):
        self.q = None
        self.delta = None
        self.omega = None
        self.cost_ISI = None

        # To do here, with a solver ... 

    def update_s_prime(self, s_prime, G=1):
        # here we change s_prime accordingly to the ISI we just computed. Maybe also do compute_r on s_prime ?? 
        pass




class Meta_param : 
    def __init__(self, eta, Delta, tau):
        self.eta = eta
        self.Delta = Delta
        self.tau = tau


class Matheuristic : 
    def __init__(self, initial_solution):
        self.operators = [
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator1, 'name': 'operator1' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator2, 'name': 'operator2' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator3, 'name': 'operator3' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':operator4 , 'name': 'operator4'}
        ]

        self.solution = initial_solution
        self.solution_best = deepcopy(initial_solution)
        self.solution_prime = deepcopy(initial_solution)


    def final_algo(self, meta_param):
        # here one can do the final matheuristic described in the paper
        pass


    def operator1(solution):
        pass

    def operator2(solution):
        pass

    def operator3(solution):
        pass

    def operator4(solution):
        pass