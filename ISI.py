import numpy as np
import numpy.random as rd
from copy import deepcopy
#edit Chris 07.06.20:
from OR_tools_solve_tsp import tsp_tour
#end of edit Chris 07.06.20

class Problem :
    #this is the class that contains the data of the problem
    def __init__(self,D,Warehouses,Schools,T,K, Q1, Q2, v, t_load, c_per_km):
        self.D = D # distance matrix. Could be a pandas data frame with the names of Warehouses/Schools as index of rows and colomns 
                    # to get the distance between a warehouse and a school for example : D.loc[warehouse_name, school_name]
        self.Warehouses = Warehouses # list of dictionary {'capacity': ..., 'lower':..., 'dist_central': ... , 'fixed_cost': ... , 'initial': ...,  'name' : ...}
        self.Schools = Schools  # list of dictionary {'capacity': ..., 'lower':..., 'consumption': ...,'storage_cost': ... , 'initial': ...,  'name' : ...}
        self.T = T # time horizon
        self.K = K # number of vehicles
        self.Q1 = Q1 # capacity of the trucks for school deliveries
        self.Q2 = Q2 # capacity of the trucks for warehouses deliveries
        self.v = v # average speed of trucks in km/h
        self.t_load = t_load # average loading/unloading time at schools in hours
        self.c_per_km = c_per_km # average routing cost per kilometer

    def define_arrays(self):
        self.I_s_init  =  np.array([s["initial"] for s in self.Schools])           # initial inventory of school
        self.U_s       =  np.array([s["capacity"] for s in self.Schools])          # capactiy upper bound school
        self.L_s       =  np.array([s["lower"] for s in self.Schools])             # capacity lower bound school     
        self.h_s       =  np.array([s["storage_cost"] for s in self.Schools])      # storage cost school
        self.d         =  np.array([s["consumption"] for s in self.Schools])       # consumption of schools

        self.I_w_init  =  np.array([w["initial"] for w in self.Warehouses])        # initial inventory of warehouses
        self.U_w       =  np.array([w["capacity"] for w in self.Warehouses])       # capactiy upper bound warehouse
        self.L_w       =  np.array([w["lower"] for w in self.Warehouses])          # capacity lower bound warehouse
        self.F_w       =  np.array([w["fixed_cost"] for w in self.Warehouses])     # capacity lower bound warehouse
        self.to_central=  np.array([w["dist_central"] for w in self.Warehouses])   # distance between the warehouses and the central





class Solution : 
    def __init__(self,problem):
        M,N,K,T = len(problem.Schools), len(problem.Warehouses),problem.K, problem.T
        self.M, self.N, self.K, self.T = M,N,K,T

        self.name_schools    = [s["name"] for s in problem.Schools ]
        self.name_warehouses = [w["name"] for w in problem.Warehouses ]
        
        problem.define_arrays()
        self.problem = problem
           
        self.Y = np.zeros((T,N,K,M), dtype = bool)      # variable  equal 1 if vehicle k delivers school m from warehouse n at time t
        self.q = np.zeros((T,N,K,M), dtype = float)     # quantity of food delivered by vehicle k delivers school m from warehouse n at time t
        
        self.X = np.zeros((T,N), dtype = bool )   # variable equal 1 if warehouse n get more food at time t
        
        
        # other variable to add probably
    def compute_route_cost(self, tour, warehouse):
        dist = self.problem.D.values
        tour2   = [warehouse]+tour+[warehouse]
        return sum( [ dist[tour2[i],tour2[i+1]] for i in range(len(tour2)-1)])
    
    def compute_a_and_b(self):
        # to do 
        self.a = np.zeros((self.T,self.N,self.K,self.M)) # routing cost reduction if school m is removed from the tour of vehicle k from WH n at time t ==> array TxKxMxN
        self.b = np.zeros((self.T,self.N,self.K,self.M)) # routing cost addition if school m is added to the tour of vehicle k from WH n at time t ==> array TxKxMxN
        dist = self.problem.D.values

        for t in range(self.T): 
            for n in range(self.N):
                for k in range(self.K):
                    tour = self.r[t][n][k]
                    tour2   = [n]+tour+[n]

                    #compute a
                    # to do : make it with vector operations

                    for i in range(1,len(tour2)-1) : 
                        self.a[t,n,k, tour2[i]-self.N] = dist[tour2[i],tour2[i+1]]+dist[tour2[i],tour2[i-1]] - dist[tour2[i-1],tour2[i+1]]
                        
                    # compute b 
                    edges_cost = np.array( [dist[tour2[i],tour2[i+1]] for i in range(len(tour2)-1)] )
                    for m in np.nonzero(1 - self.Y[t,n,k,:])[0]:
                        add_edges_cost =  np.array( [ dist[m+self.N,tour2[i]]+dist[m+self.N,tour2[i+1]]  for i in range(len(tour2)-1) ] )
                        self.b[t,n,k,m] = np.amin(add_edges_cost-edges_cost)
                   
    def compute_r(self):
        # here are the TSP to be computed
        self.r = [[[[] for i in range(self.K)] for i in range(self.N)] for i in range(self.T)] # for each time t, for each vehicle k, a list of ordered integer of [0, M+N] corresponding to a tour
        dist = self.problem.D.values
        for t in range(self.T):
            for n in range(self.N):
                for k in range(self.K):
                    tour = np.nonzero(self.Y[t,n,k,:])[0] + self.N 
                    #tour = [n] + np.ndarray.tolist(np.array(s[0])+self.N) + [n] #  the tour starts at the warehouse then add the school in the wrong order
                    #edit Chris 07.06.20:
                    tour = tsp_tour(tour, n, dist)  #function returns optimal tour and length of optimal tour
                    #end of edit Chris 07.06.20
                    self.r[t][n][k] = tour

    def compute_transport_costs(self):
        return np.array([[[
                        self.compute_route_cost(self.r[t][n][k],n) for k in range(self.K)
                        ] for n in range(self.N)
                        ] for t in range(self.T)
                    ])

    def compute_costs(self, add = 0): 
        self.cost = self.compute_transport_costs() + add
        
    def compute_time_adding(self):
        problem  = self.problem

        lenghts = np.array( [[[ len(self.r[t][n][k]) for k in range(self.K) ] for n in range(self.N) ] for  t in range(self.T) ]   )
        time_route = self.cost / problem.v + problem.t_load*lenghts
        time_adding = np.array( [ self.b[:,:,:,m] / problem.v + time_route + problem.t_load  for m in range(self.M) ]  )
        np.swapaxes(time_adding,0,1)
        np.swapaxes(time_adding,1,2)
        np.swapaxes(time_adding,2,3)
        # now time_adding is a matrixe of size TxNxKxM
        self.time_adding = time_adding


    def ISI(self, G = 1):
        # change the solution itself to the ISI solution
        problem = self.problem

        self.compute_a_and_b()

        self.compute_time_adding()
        
        # just to remember : the psi of the paper is the same thing as our Y

        



        '''
        Decision variables : 
        q          variable of size TxNxKxM, type= positive float representing how much food to deliver at each stop of a truck
        X          variable of size TxN type = bool representing the pick ups  
        delta      variable of size TxNxKxM, type=bool representing wether or not l is removed from the tour
        omega      variable of size TxNxKxM, type=bool representing wether or not l is added to the tour
        '''

        '''
        Other variables : 

        # variable of size TxM type = float representing the inventories of the schools
        I_s[0,:] = problem.I_s_init[:] 
        for t in range(self.T):       
            I_s[t,:] = I_s[t-1,:] + problem.Q1 * sum(q[t,:,:,:], axis = 0,1 ) - d

        # variable of size TxM type = float representing the inventories of the warehouses        
        I_w[0,:] = problem.I_w_init[:]    
        for t in range(self.T):    
            I_w[t,:] = I_w[t-1,:] + problem.Q2 * X[t,:] - problem.Q1* sum( q[t,:,k,l] axis = 1,2 )

        '''


        '''
        Constraints : 
        # constraints 14 to 17 are omitted here (according to Milena's script)

        omega[t,n,k,m]*self.time_adding[t,n,k,m]  < Tmax
        I_s < problem.U_s
        I_s > problem.L_s
        I_w < problem.U_w      # can maybe be omitted 
        I_w > problem.L_w
        sum(q, axis = 3) < 1
        # constraint 10 omitted for now : let's consider that at time t, delivering is after lunch, and L[l] > d[l] for every school l 
        q < (self.Y - delta + omega)   # no need to multiply by U because the componant of q is already smaller than 1 because it is normalized by Q1
        omega < 1 - self.Y
        delta < self.Y
        sum(delta+omega, axis = 3) < G
        '''

        '''
        Objective function : 
        minimize : 
                sum( problem.h_s * sum(I_s,axis=0) ) 
            +   problem.c_per_km * sum( problem.to_central * sum(X,axis=0)  ) * 2
            +   problem.c_per_km * sum(self.b*omega, axis=all)
            -   problem.c_per_km * sum(self.a*delta, axis=all)

        '''

        '''
        Cost except for omega and delta, so some part of the objective function :
        add_cost = sum( problem.h_s * sum(I_s,axis=0) ) 
            +   2* sum( problem.to_central * sum(r,axis=0)  )
        '''


        self.update_after_ISI(delta,omega,q, X)
        self.compute_r()
        self.compute_costs(add=add_cost)


    def update_after_ISI(self,delta,omega,q, X): 

        self.Y = self.Y + omega - delta

        self.X = X[:,:]
        self.q = self.problem.Q1*q
        
        # probably se




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
        self.rho = 10


class Matheuristic : 
    def __init__(self, initial_solution):
        self.operators = [
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.rand_remove_rho, 'name': 'rand_remove_rho' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.remove_worst_rho, 'name': 'remove_worst_rho' },
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
            operator(self.solution_prime, param.rho)
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

        

                





    def rand_remove_rho(solution, rho):
        Y_flat = solution.Y.reshape(-1)
        num_served = len(Y_flat)
        rho_samples = np.random.choice(num_served, rho, replace = False)
        Y_flat[rho_samples] = 0

    def remove_worst_rho(solution, rho):
        a_flat = solution.a.reshape(-1)
        Y_flat = solution.Y.reshape(-1)
        Y_flat[np.argpartition(-a_flat, rho)[0:rho]] = 0

    def operator3(solution):
        pass

    def operator4(solution):
        pass


    def choose_operator(operators):
        weights = [operator['weight'] for operator in operators]
        s = 0.
        v = rd.random()*sum(weights)
        for i in range(len(weights)):
            s+=weights[i]
            if s>=v : return i


    def update_weights(self, r):        # r is the reaction factor
        for op in self.operators : 
            if (op['number_used']>0): op['weight'] = (1-r)*op['weight'] + r* op['score']/op['number_used']
            op['score']  = 0
            op['number_used'] = 0





'''
TO DO LIST : 

- Do a fake ISI function to test our Matheuristic
- Find how to do the TSPs, and compute_r
- Write the ISI function
- Write the operators
- choose_operator
- compute_a_and_b
'''

# test !




