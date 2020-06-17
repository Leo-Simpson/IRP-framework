import numpy as np
import numpy.random as rd
import random
from copy import deepcopy
#edit Chris 07.06.20:
from OR_tools_solve_tsp import tsp_tour
#end of edit Chris 07.06.20
import pulp as plp 

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
        self.F_w       =  np.array([w["fixed_cost"] for w in self.Warehouses])     # fixed costs for each warehouse
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
        self.q = np.zeros((T,N,K,M), dtype = float)     # quantity of food delivered from each warehouse n by vehicle k delivers school m at time t
        
        self.X = np.zeros((T,N), dtype = bool )   # variable equal 1 if warehouse n get more food at time t

        self.Cl = np.ones((N,M),dtype=bool)    # equal one when we consider that it is possible that the school m could be served by n
    
    def copy(self):
        solution = Solution(self.problem)
        

        # other variable to add probably
    
    def Cl_shaped_like_Y(self):
        Cl_times_K = np.repeat(self.Cl[np.newaxis,0,:], self.K, 0)[np.newaxis,:,:]
        for i in np.arange(1,self.N):
            Cl_times_K = np.concatenate([Cl_times_K, np.repeat(self.Cl[np.newaxis,i,:], self.K, 0)[np.newaxis,:,:]],0)
        return np.repeat(Cl_times_K[np.newaxis,:,:,:], self.T, 0)
    
    def compute_school_remove(self,t,n,k):
        
        dist_mat = self.problem.D.values
        tour_school = self.r[t][n][k]
        tour_complete = [n]+tour_school+[n] 
        for i in range(1,len(tour_complete)-1): 
            self.a[t,n,k, tour_complete[i] - self.N] = dist_mat[tour_complete[i], tour_complete[i+1]] + dist_mat[tour_complete[i], tour_complete[i-1]] - dist_mat[tour_complete[i-1], tour_complete[i+1]]
            
    def compute_school_insert(self,t,n,k):
        
        dist_mat = self.problem.D.values
        tour_school = self.r[t][n][k]
        tour_complete   = [n]+tour_school+[n] 
        edges_cost = np.array( [dist_mat[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)] )
        allowed = [m for m in np.nonzero(1 - self.Y[t,n,k,:])[0] if self.Cl[n,m] == 1]
        for m in allowed:
            add_edges_cost =  np.array( [ dist_mat[m+self.N,tour_complete[i]]+dist_mat[m+self.N,tour_complete[i+1]]  for i in range(len(tour_complete)-1) ] )
            self.b[t,n,k,m] = np.amin(add_edges_cost-edges_cost)
    
    def compute_a_and_b(self):
       
        self.a = np.zeros((self.T,self.N,self.K,self.M)) # routing cost reduction if school m is removed from the tour of vehicle k from WH n at time t ==> array TxKxMxN
        self.b = np.zeros((self.T,self.N,self.K,self.M)) # routing cost addition if school m is added to the tour of vehicle k from WH n at time t ==> array TxKxMxN

        for t in range(self.T): 
            for n in range(self.N):
                for k in range(self.K):
                    
                    self.compute_school_remove(t,n,k)
                        
                    self.compute_school_insert(t,n,k)
                   
    def compute_r(self):
        # here are the TSP to be computed
        self.r = [[[[] for k in range(self.K)] for n in range(self.N)] for t in range(self.T)] # for each time t, for each vehicle k, for each warehouse n, a list of ordered integer of [0, M+N] corresponding to a tour
        dist_mat = self.problem.D.values
        for t in range(self.T):
            for n in range(self.N):
                for k in range(self.K):
                    tour_school = np.nonzero(self.Y[t,n,k,:])[0] + self.N 
                    #tour = [n] + np.ndarray.tolist(np.array(s[0])+self.N) + [n] #  the tour starts at the warehouse then add the school in the wrong order
                    #edit Chris 07.06.20:
                    tour = tsp_tour(tour_school, n, dist_mat)  #function returns optimal tour and length of optimal tour
                    #end of edit Chris 07.06.20
                    self.r[t][n][k] = tour    # is this tour with or without the warehouse ?? It should be without

    def compute_dist(self):
        self.dist = np.array([[[
                        self.compute_route_dist(self.r[t][n][k],n) for k in range(self.K)
                        ] for n in range(self.N)
                        ] for t in range(self.T)
                    ])

    def compute_cost(self, add = 0): 
        self.compute_dist()
        self.cost = self.problem.c_per_km * self.dist + add

    def compute_route_dist(self, tour_schools, warehouse : int):
        dist_mat = self.problem.D.values
        tour_complete   = [warehouse]+tour_schools+[warehouse]
        return sum( [ dist_mat[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)])

    def compute_time_adding(self):
        problem  = self.problem

        lenghts = np.array( [[[ len(self.r[t][n][k]) for k in range(self.K) ] for n in range(self.N) ] for  t in range(self.T) ]   )
        time_route = self.dist / problem.v + problem.t_load*lenghts
        time_adding = np.array( [ self.b[:,:,:,m] / problem.v + problem.t_load  for m in range(self.M) ]  )
        np.swapaxes(time_adding,0,1)
        np.swapaxes(time_adding,1,2)
        np.swapaxes(time_adding,2,3)
        # now time_adding is a matrixe of size TxNxKxM
        self.time_adding = time_adding
        self.time_route = time_route


    def ISI(self, G = 1):
        # change the solution itself to the ISI solution
        problem = self.problem
        T,N,K,M = self.T, self.N, self.K, self.M

        self.compute_a_and_b()

        self.compute_time_adding()

        set_q     = [ (t,n,k,m) for t in range(T) for n in range(N) for k in range(K) for m in range(M) if self.Cl[n,m]  ]
        set_delta = [ (t,n,k,m) for (t,n,k,m) in set_q if self.Y[t,n,k,m]  ]
        set_omega = [ (t,n,k,m) for (t,n,k,m) in set_q if not self.Y[t,n,k,m]  ]


        ISI_model=plp.LpProblem(plp.LpMinimize,name="ISI_Model")

        # build dictionaries of decision variables:
        q_vars = plp.LpVariable.dicts("q",set_q, cat='Continuous', lowBound=0., upBound=1.)
        X_vars = plp.LpVariable.dicts("X",[(t,n) for t in range(T) for n in range(N)], cat='Binary')
        delta_vars = plp.LpVariable.dicts("delta",set_delta, cat='Binary')
        omega_vars = plp.LpVariable.dicts("omega",set_omega, cat='Binary')
        
        
        # just to remember : the psi of the paper is the same thing as our Y
        

        I_s = {(0,m): problem.I_s_init[n]   for m in range(M) }   # need to see how to change an LpAffineExpression with a constant value
        I_w = {(0,n): problem.I_w_init[n]   for n in range(N) }  # need to see how to change an LpAffineExpression with a constant value

        for t in range (1,T): 
            I_s.update(  {(t,m):
                         I_s[t-1,m]
                         + problem.Q1 * plp.lpSum(q_vars[t,n,k,m] for k in range(K) for n in range(N) if self.Cl[n,m] ) 
                         - problem.d[m]
                         for m in range(M) }  
                        )
        
            I_w_init.update(  {(t,n):
                         I_w[t-1,m]  
                         - problem.Q1 * plp.lpSum(q_vars[t,n,k,m] for k in range(K) for m in range(m) if self.Cl[n,m] ) 
                         + problem.Q2 * X_vars[t,n]
                         for n in range(N) }  
                        )


        ISI_model += plp.lpSum( problem.h_s[m] * I_s[t,n] for t in range(T) for m in range(m) )
        + problem.c_per_km * plp.lpSum( problem.to_central[n] * X_vars[t,n] for t in range(T) for n in range(n) ) 
        + problem.c_per_km * plp.lpSum( problem.b[t,n,k,m] * omega_vars[t,n,k,m] for (t,n,k,m) in set_omega )
        - problem.c_per_km * plp.lpSum( problem.a[t,n,k,m] * delta_vars[t,n,k,m] for (t,n,k,m) in set_delta ), 'Z'

        # constraint 9 in Latex script, respect capacities + min. stock of schools and warehouses
        
        for t in range(T):
            # schools: problem.L_s < I_s < problem.U_s
            for m in range(M):
                ISI_model += I_s[t,m]<=problem.U_s[m]       #I_s < U_s
                ISI_model += I_s[t,m]>= problem.L_S[m]      #I_s > L_s
            # warehouses: problem.L_w <I_w < problem.U_w
            for n in range(N):
                ISI_model += I_w[t,n]<=problem.U_w[n]       #I_w < U_w      # can maybe be omitted 
                ISI_model += I_s[t,n]>= problem.L_S[n]      #I_w > L_w

        # constraint on capacity of trucks
        #sum(q, axis = 3) < 1
        for t in range(T):
            for n in range(N):
                for k in range(K):
                    ISI_model += plp.lpSum([q_vars[t][n][k][m] for  m in range(M)]) <=1

        
        # constraint 11: only positive amount to deliver if school is served in that round
        #q < (self.Y - delta + omega)   # no need to multiply by U because the component of q is already smaller than 1 because it is normalized by Q1
        for (t,n,k,m) in set_q:     
            if (t,n,k,m) in set_delta :
                ISI_model += q_vars[t,n,k,m] <= 1 - delta_vars[t,n,k,m]
            else : 
                ISI_model += q_vars[t,n,k,m] <= omega_vars[t,n,k,m]

        #constraint 18: bound on the number of changes comitted by the ISI model
        #sum(delta+omega, axis = 3) < G
        for t in range(T):
            for k in range(K):
                ISI_model += plp.Lpsum([delta_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_delta ]) + plp.Lpsum([omega_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_omega ] )


        # We only need the Tmax thing to write 



        ISI_model.solve()

        # transform the _vars things into numpy array to return it. 

        # evaluate the I_s and I_w to be able to build add_cost


        '''


        Decision variables : 
        q          variable of size TxNxKxM, type= positive float representing how much food (actually: percentage of truck load) to deliver at each stop of a truck
        X          variable of size TxN type = bool representing the pick ups  
        delta      variable of size TxNxKxM, type=bool representing wether or not l is removed from the tour
        omega      variable of size TxNxKxM, type=bool representing wether or not l is added to the tour

        Other variables : 

        # variable of size TxM type = float representing the inventories of the schools
        I_s[0,:] = problem.I_s_init[:] 
        for t in range(self.T):       
            I_s[t,:] = I_s[t-1,:] + problem.Q1 * sum(q[t,:,:,:], axis = 0,1 ) - problem.d

        # variable of size TxM type = float representing the inventories of the warehouses        
        I_w[0,:] = problem.I_w_init[:]    
        for t in range(self.T):    
            I_w[t,:] = I_w[t-1,:] + problem.Q2 * X[t,:] - problem.Q1* sum( q[t,:,k,l] axis = 1,2 )


        Objective function : 
        minimize : 
                sum( problem.h_s * sum(I_s,axis=0) ) 
            +   problem.c_per_km * sum( problem.to_central * sum(X,axis=0)  ) * 2
            +   problem.c_per_km * sum(self.b*omega, axis=all)
            -   problem.c_per_km * sum(self.a*delta, axis=all)


        Constraints : 
        # constraints 14 to 17 are omitted here (according to ShareLatex script)
        
        # constraint on length of tour
        #sum( omega*self.time_adding, axis  = 3 ) + self.time_route - sum( delta*self.time_substracting, axis  = 3 )  < Tmax
        for t in range(T):
            for n in range(N):
                for k in range(K):
                    ISI_model += plp.lpSum([omega_vars[t][n][k][m] * self.time_adding[t,n,k,m] +  .... for m in .... ]) <= Tmax

        # constraint 9 in Latex script, respect capacities + min. stock of schools and warehouses
        # schools: problem.L_s < I_s < problem.U_s
        for t in range(T):
            for m in range(M):
                ISI_model += I_s[t,m]<=problem.U_s[m]       #I_s < U_s
                ISI_model += I_s[t,m]>= problem.L_S[m]      #I_s > L_s
        # warehouses: problem.L_w <I_w < problem.U_w
            for n in range(N):
                ISI_model += I_w[t,n]<=problem.U_w[n]       #I_w < U_w      # can maybe be omitted 
                ISI_model += I_s[t,n]>= problem.L_S[n]      #I_w > L_w
        
        # constraint on capacity of trucks
        #sum(q, axis = 3) < 1
        for t in range(T):
            for n in range(N):
                for k in range(K):
                    ISI_model += plp.lpSum([q_vars[t,n,k,m] for  m in range(M)]) <=1
        

        # constraint 10 omitted for now : let's consider that at time t, delivering is after lunch, and L[l] > d[l] for every school l 
        
        # constraint 11: only positive amount to deliver if school is served in that round
        #q < (self.Y - delta + omega)   # no need to multiply by U because the component of q is already smaller than 1 because it is normalized by Q1
        
        in set_delta : 
        q < 1 - delta

        in set_omega : 
        q < omega



        UNECESSARY
        for (t,n,k,m) in set_q:     
            #if t>0:        ADD: t>0?!
            ISI_model += q_vars[t][n][k][m] <= problem.U_s[m] - I_s[t-1,m]
        
        # constraint 12: school insertion only possible if not yet in route
        UNECESSARY
        #omega < 1 - self.Y
        for (t,n,k,m) in set_omega:
            ISI_model += omega_vars[t][n][k][m] <= 1 - solution.Y[t][n][k][m]


        #constraint 13: school removal only if in route
        UNECESSARY
        #delta < self.Y
        for (t,n,k,m) in set_delta:
            ISI_model += delta_vars[t][n][k][m] <= solution.Y[t][n][k][m]

        #constraint 18: bound on the number of changes comitted by the ISI model
        #sum(delta+omega, axis = 3) < G
        for t in range(T):
            for k in range(K):
                ISI_model += plp.Lpsum([delta_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_delta ]) + plp.Lpsum([omega_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_omega ] )

        '''

        '''
        Cost except for omega and delta, so some part of the objective function :
        add_cost = sum( problem.h_s * sum(I_s,axis=0) ) 
            +   2*problem.c_per_km* sum( problem.to_central * sum(r,axis=0)  )
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
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.shaw_removal_route_based, 'name': 'shaw_removal_route_based' },
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.shaw_removal_greedy, 'name': 'shaw_removal_greedy'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.avoid_consecutive_visits, 'name': 'avoid_consecutive_visits'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.empty_one_period, 'name': 'empty_one_period'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.empty_one_vehicle, 'name': 'empty_one_vehicle'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.empty_one_plant, 'name': 'empty_one_plant'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.rand_insert_rho, 'name': 'rand_insert_rho'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.assign_to_nearest_plant, 'name': 'assign_to_nearest_plant'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.insert_best_rho, 'name': 'insert_best_rho'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.swap_roh_cust_intra_routes, 'name': 'swap_roh_cust_intra_routes'},
                {'weight' : 1, 'score': 0 , 'number_used':0, 'function':Matheuristic.swap_roh_cust_intra_plants, 'name': 'swap_roh_cust_intra_plants'}
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

                keep_going, i  = True, 0
                while keep_going and i < 1000: 
                    i+=1                              #line 10
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
        served = np.nonzero(Y_flat)[0]
        num_served = len(served)
        assert rho <= num_served
        rho_samples = np.random.choice(num_served, rho, replace = False)
        Y_flat[served[rho_samples]] = 0

    def remove_worst_rho(solution, rho):
        for i in range(np.min([rho,len(np.nonzero(solution.Y)[0])])):   
            choice = np.argmax(solution.a)
            solution.Y.reshape(-1)[choice] = 0
            solution.a.reshape(-1)[choice] = 0
            t, rest = np.divmod(choice, solution.N*solution.K*solution.M)
            n, rest = np.divmod(rest, solution.K*solution.M)
            k = np.floor_divide(rest, solution.M)
            tour_school = np.nonzero(solution.Y[t,n,k,:])[0] + solution.N 
            solution.r[t][n][k] = tsp_tour(tour_school, n, solution.problem.D.values)
            solution.compute_school_remove(t,n,k)

    def shaw_removal_route_based(solution, rho):
        served = np.transpose(np.nonzero(solution.Y))
        num_served = len(served)
        candidate = np.random.choice(num_served)
        [t,n,k,m] = served[candidate]
        route = np.array(solution.r[t][n][k])
        if len(route) > 2:
            schools = route[np.where(route != m + solution.N)[0]]
            dist_from_m = solution.problem.D.values[np.ix_([m + solution.N],route)][0]
            min_dist_from_m = np.min(solution.problem.D.values[np.ix_([m + solution.N],schools)][0])
            to_remove = route[np.where(dist_from_m <= 2*min_dist_from_m)[0]] - solution.N 
            solution.Y[t,n,k,to_remove] = 0
        else:
            solution.Y[t,n,k,:] = 0


    def shaw_removal_greedy(solution, rho):
        served = np.transpose(np.nonzero(solution.Y))
        num_served = len(served)
        candidate = np.random.choice(num_served, 1)[0]
        [t,n,k,m] = served[candidate]
        route = np.array(solution.r[t][n][k])
        if len(route) > 2:
            schools = route[np.where(route != m + solution.N)[0]]
            dist_from_m = solution.problem.D.values[np.ix_([m + solution.N],route)][0]
            dist = solution.problem.D.values[np.ix_(route,route)]
            min_dist = np.min(dist[dist>0])
            to_remove = route[np.where(dist_from_m <= 2*min_dist)[0]] - solution.N 
            solution.Y[t,n,k,to_remove] = 0
        else:
            solution.Y[t,n,k,:] = 0
    
    def avoid_consecutive_visits(solution, rho):
        for t in range(solution.T-1):
            time_schools = np.sum(solution.Y[[t, t+1],:,:,:], axis = (1,2))
            print(time_schools)
            index = np.where(time_schools[1,:] + time_schools[0,:] > 1)
            solution.Y[t+1,:,:,index] = 0
    
    def empty_one_period(solution, rho):
        period = np.random.randint(solution.T)
        solution.Y[period,:,:,:] = 0
    
    def empty_one_vehicle(solution, rho):
        warehouse = np.random.randint(solution.N)
        vehicle = np.random.randint(solution.K)
        solution.Y[:,warehouse,vehicle,:] = 0
    
    def empty_one_plant(solution, rho):
        warehouse = np.random.randint(solution.N)
        solution.Y[:,warehouse,:,:] = 0
        
    def furthest_customer(solution, rho):
        for i in range(rho):
            t,n,k = np.random.randint(solution.T), np.random.randint(solution.N), np.random.randint(solution.K)
            route = solution.r[t][n][k]
            furthest_customer = route[np.argmax(solution.problem.D.values[np.ix_([n],route)][0])] - solution.N
            solution.Y[t,n,k,furthest_customer] = 0
        
    def rand_insert_rho(solution, rho):    
        for i in range(rho):
            t,n,k = np.random.randint(solution.T), np.random.randint(solution.N), np.random.randint(solution.K)
            not_served_and_allowed = np.where(np.sum(solution.Y[t,:,:,:], axis = (0,1)) + 1 - solution.Cl[n,:] == 0)[0] 
            if len(not_served_and_allowed) > 0:
                m = np.random.choice(not_served_and_allowed)
                solution.Y[t,n,k,m] = 1
        
    def assign_to_nearest_plant(solution, rho):
        for i in range(rho):
            t = np.random.randint(solution.T)
            not_served = np.where(np.sum(solution.Y[t,:,:,:], axis = (0,1)) == 0)[0]
            if len(not_served) > 0:
                m = np.random.choice(not_served)
                plants = [i for i in range(solution.N) if solution.Cl[i,m] == 1]
                nearest_plant = plants[np.argmin(solution.problem.D.values[np.ix_([m + solution.N],plants)][0])]
                solution.Y[t,nearest_plant,0,m] = 1    #to do: shouldn't be k == 0. Change to vehicle with least  insertion cost.
            else:
                pass
        
    def insert_best_rho(solution, rho):
        for i in range(np.min([rho, len(np.where(solution.Y + 1 - solution.Cl_shaped_like_Y()  == 0)[0])])):   
            b_flat = solution.b.reshape(-1)
            Y_flat = solution.Y.reshape(-1)
            allowed = Y_flat + 1 - solution.Cl_shaped_like_Y().reshape(-1)  == 0
            choice = np.where(allowed)[0][np.argmin(b_flat[allowed])]
            Y_flat[choice] = 1
            b_flat[choice] = 0
            t, rest = np.divmod(choice, solution.N*solution.K*solution.M)
            n, rest = np.divmod(rest, solution.K*solution.M)
            k = np.floor_divide(rest, solution.M)
            tour_school = np.nonzero(solution.Y[t,n,k,:])[0] + solution.N 
            solution.r[t][n][k] = tsp_tour(tour_school, n, solution.problem.D.values)
            solution.compute_school_insert(t,n,k)
        
    def shaw_insertion(solution, rho):  #to do: what to do if there is no vehicle available? and if not all chosen customers can be served (variable Cl) by the closest warehouse?
        period = np.random.randint(solution.T)
        not_served = np.where(np.sum(solution.Y[period,:,:,:], axis = (0,1)) == 0)[0]
        (index, choice) = random.choice(list(enumerate(not_served)))
        dist_to_all = solution.problem.D.values[np.ix_([choice + solution.N],[m  + solution.N for m in range(solution.M) if m != choice])][0]
        rest_not_served = np.delete(not_served, index)
        dist_to_not_served = solution.problem.D.values[np.ix_([choice + solution.N],rest_not_served + solution.N)][0]
        close = rest_not_served[dist_to_not_served <= 2*np.min(dist_to_all)]
        closest_warehouse = np.argmin(solution.problem.D.values[np.ix_([choice + solution.N],[i for i in range(solution.N)])][0])
        close_reachable = [m for m in close[solution.Cl[closest_warehouse, close] == 1]]
        free_vehicles = np.where(np.sum(solution.Y[period,closest_warehouse,:,:], axis = 1) == 0)[0]
        if len(free_vehicles) > 0:
            solution.Y[period,closest_warehouse,np.min(free_vehicles), close_reachable] = 1
        
    def swap_roh_cust_intra_routes(solution, rho):
        for i in range(rho):
            period = np.random.randint(solution.T)
            warehouse = np.random.randint(solution.N)
            if len(np.where(np.sum(solution.Y[period,warehouse,:,:], axis = 1) > 0)[0]) >=2:
                vehicle1, vehicle2 = np.random.choice(np.where(np.sum(solution.Y[period,warehouse,:,:], axis = 1) > 0)[0], 2, replace = False)
                school1, school2 = np.random.choice(np.where(solution.Y[period,warehouse,vehicle1,:] == 1)[0]), np.random.choice(np.where(solution.Y[period,warehouse,vehicle2,:] == 1)[0])
                solution.Y[period,warehouse,[vehicle1, vehicle2],school1] = [0, 1]
                solution.Y[period,warehouse,[vehicle1, vehicle2],school2] = [1, 0]
        
    def swap_roh_cust_intra_plants(solution, rho):
        for i in range(rho):
            period = np.random.randint(solution.T)
            warehouse1, warehouse2 = np.random.choice([n for n in range(solution.N) if np.sum(solution.Y[period, n, :, :]) > 0], 2, replace = False)
            school1, school2 = np.random.choice(np.where(np.sum(solution.Y[period,warehouse1,:,:], axis = 0) > 0)[0]), np.random.choice(np.where(np.sum(solution.Y[period,warehouse2,:,:], axis = 0) > 0)[0])
            if solution.Cl[warehouse1, school2] == 1 and solution.Cl[warehouse2, school1] == 1:
                vehicle1, vehicle2 = np.where(solution.Y[period,warehouse1,:,school1] > 0)[0], np.where(solution.Y[period,warehouse2,:,school2] > 0)[0]
                solution.Y[period,warehouse1,vehicle1,school1] = 0
                solution.Y[period,warehouse2,vehicle2,school1] = 1
                solution.Y[period,warehouse1,vehicle1,school2] = 1
                solution.Y[period,warehouse2,vehicle2,school2] = 0

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







# test !




