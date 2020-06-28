import numpy as np
import numpy.random as rd
import random
from copy import deepcopy
from OR_tools_solve_tsp import tsp_tour
import pulp as plp 
from scipy.spatial import distance_matrix
import pandas as pd

import plotly.graph_objects as go 
from plotly import offline
from visu import visu 


class Problem :
    #this is the class that contains the data of the problem
    def __init__(self,Warehouses,Schools,T,K, Q1, Q2, v, t_load, c_per_km, Tmax, central = None, D = None):
        if central is None : self.central = np.zeros(2)
        else: self.central = central

        central_w = {"capacity": np.inf, "lower":-np.inf, "dist_central":0, "fixed_cost":0, "initial": 0, "name": "CENTRAL" }
        self.Warehouses = [central_w] + Warehouses # list of dictionary {'capacity': ..., 'lower':..., 'dist_central': ... , 'fixed_cost': ... , 'initial': ...,  'name' : ...}
        
        
        self.Schools = Schools  # list of dictionary {'capacity': ..., 'lower':..., 'consumption': ...,'storage_cost': ... , 'initial': ...,  'name' : ...}
        self.T = T # time horizon
        self.K = K # number of vehicles
        self.Q1 = Q1 # capacity of the trucks for school deliveries
        self.Q2 = Q2 # capacity of the trucks for warehouses deliveries
        self.v = v # average speed of trucks in km/h
        self.t_load = t_load # average loading/unloading time at schools in hours
        self.c_per_km = c_per_km # average routing cost per kilometer
        self.Tmax = Tmax

        if D is None : 
            locations = [w['location'] for w in self.Warehouses] + [s['location'] for s in self.Schools] 
            names = [w['name'] for w in self.warehouses] + [s['name'] for s in self.schools]
            self.D = pd.DataFrame(distance_matrix(locations,locations), columns = names, index=names)
        else : 
            self.D = D # distance matrix. Could be a pandas data frame with the names of Warehouses/Schools as index of rows and colomns 
            # to get the distance between a warehouse and a school for example : D.loc[warehouse_name, school_name]

        for w in self.Warehouses : 
            if w["dist_central"]<=0. : w["dist_central"] = self.D.loc("CENTRAL",w["name"])*2

        

    def define_arrays(self):
        self.I_s_init  =  np.array([s["initial"] for s in self.Schools])           # initial inventory of school
        self.U_s       =  np.array([s["capacity"] for s in self.Schools])          # capactiy upper bound school
        self.L_s       =  np.array([s["lower"] for s in self.Schools])             # capacity lower bound school     
        self.h_s       =  np.array([s["storage_cost"] for s in self.Schools])      # storage cost school
        self.d         =  np.array([s["consumption"] for s in self.Schools])       # consumption of schools
        self.dt        =  np.array([self.d]*self.T)

        self.I_w_init  =  np.array([w["initial"] for w in self.Warehouses])        # initial inventory of warehouses
        self.U_w       =  np.array([w["capacity"] for w in self.Warehouses])       # capactiy upper bound warehouse
        self.L_w       =  np.array([w["lower"] for w in self.Warehouses])          # capacity lower bound warehouse
        self.F_w       =  np.array([w["fixed_cost"] for w in self.Warehouses])     # fixed costs for each warehouse
        self.to_central=  np.array([w["dist_central"] for w in self.Warehouses])   # distance between the warehouses and the central





class Solution : 
    
    def __init__(self,problem, Y = None, q = None, X = None, Cl=None):
        M,N,K,T = len(problem.Schools), len(problem.Warehouses),problem.K, problem.T
        self.M, self.N, self.K, self.T = M,N,K,T

        self.name_schools    = [s["name"] for s in problem.Schools ]
        self.name_warehouses = [w["name"] for w in problem.Warehouses ]
        
        problem.define_arrays()
        self.problem = problem
           
        if Y is None : self.Y = np.zeros((T,N,K,M), dtype = bool)      # variable  equal 1 if vehicle k delivers school m from warehouse n at time t
        else         : self.Y = Y
        if q is None : self.q = np.zeros((T,N,K,M), dtype = float)     # quantity of food delivered from each warehouse n by vehicle k delivers school m at time t
        else         : self.q = q

        if X is None : self.X = np.zeros((T,N), dtype = bool )   # variable equal 1 if warehouse n get more food at time t
        else         : self.X = X

        if Cl is None : self.Cl = np.ones((N,M),dtype=bool)    # equal one when we consider that it is possible that the school m could be served by n
    
        self.r = [[[[] for k in range(self.K)] for n in range(self.N)] for t in range(self.T)]

        self.build_Cl()


    def copy(self):
        solution = Solution(self.problem,
                             Y = np.copy(self.Y),
                             q = np.copy(self.q),
                             Cl= self.Cl      )

        solution.r = deepcopy(self.r)
        solution.a = np.copy(self.a)
        solution.b = np.copy(self.b)
        solution.dist = np.copy(self.dist)
        solution.cost = self.cost

        return solution

        
    def build_Cl(self):
        for m in range(self.M):
            #dist_vect = np.array( [ self.problem.D.loc[self.name_warehouses[i], self.name_schools[m]] for i in range(self.N) ] )   # very ugly way to write it ....
            dist_vec = self.problem.D[self.name_schools[m]].values[:self.N]

            dist_max = (self.problem.Tmax - self.problem.t_load) * self.problem.v /2          #only warehouses allowed to serve schools that are reachable in tour within Tmax (   
            dist_radius = 2*np.min(dist_vect)                                               #alternatively: only warehouses allowed that are not more than twice (or take another value) far away as the closest warehouses
            dist_max = min(dist_max,dist_radius)
            self.Cl[dist_vect > dist_max , m] = False

    
    def compute_school_remove_costs(self,t,n,k):
        
        dist_mat = self.problem.D.values
        tour_complete = [n]+self.r[t][n][k]+[n]
        for i in range(1,len(tour_complete)-1): 
            self.a[t,n,k, tour_complete[i] - self.N] = dist_mat[tour_complete[i], tour_complete[i+1]] + dist_mat[tour_complete[i], tour_complete[i-1]] - dist_mat[tour_complete[i-1], tour_complete[i+1]]
            
    def compute_school_insert_costs(self,t,n,k):
        
        dist_mat = self.problem.D.values
        tour_complete   = [n]+self.r[t][n][k]+[n] 

        edges_cost = np.array( [dist_mat[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)] )
        allowed  = [m for m in range(self.M) if self.Cl[n,m] and not self.Y[t,n,k,m] ]
        #allowed = [m for m in np.where(np.sum(self.Y[t,:,:,:], axis = (0,1)) == 0)[0] if self.Cl[n,m] == 1]
        for m in allowed:
            add_edges_cost =  np.array( [ dist_mat[m+self.N,tour_complete[i]]+dist_mat[m+self.N,tour_complete[i+1]]  for i in range(len(tour_complete)-1) ] )
            self.b[t,n,k,m] = np.amin(add_edges_cost-edges_cost)
            
    def cheapest_school_insert(self,t,n,k,m):
        
        dist_mat = self.problem.D.values
        tour_school = self.r[t][n][k]
        tour_complete   = [n]+tour_school+[n] 
        edges_cost = np.array( [dist_mat[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)] )
        add_edges_cost =  np.array( [ dist_mat[m+self.N,tour_complete[i]]+dist_mat[m+self.N,tour_complete[i+1]]  for i in range(len(tour_complete)-1) ] )
        position = np.argmin(add_edges_cost)
        cost = add_edges_cost[position]
        return tour_school[:position] + [m] + tour_school[position:], cost
    
    def compute_a_and_b(self):
       
        self.a = np.zeros((self.T,self.N,self.K,self.M)) # routing cost reduction if school m is removed from the tour of vehicle k from WH n at time t ==> array TxKxMxN
        self.b = np.zeros((self.T,self.N,self.K,self.M)) # routing cost addition if school m is added to the tour of vehicle k from WH n at time t ==> array TxKxMxN

        for t in range(self.T): 
            for n in range(self.N):
                for k in range(self.K):
                    
                    self.compute_school_remove_costs(t,n,k)
                        
                    self.compute_school_insert_costs(t,n,k)
                   
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
                    self.r[t][n][k] = [i-self.N for i in tour]    # is this tour with or without the warehouse ?? It should be without

    def compute_dist(self):
        self.dist = np.array([[[
                        self.compute_route_dist(self.r[t][n][k],n) for k in range(self.K)
                        ] for n in range(self.N)
                        ] for t in range(self.T)
                    ])

    def compute_costs(self, add = 0): 
        self.compute_dist()
        if not add is None : self.cost = self.problem.c_per_km * np.sum(self.dist) + add
        else : self.cost = self.problem.c_per_km * np.sum(self.dist)

    def compute_route_dist(self, tour_schools, warehouse : int):
        dist_mat = self.problem.D.values
        tour_complete   = [warehouse]+tour_schools+[warehouse]
        return sum( [ dist_mat[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)])

    def compute_time_adding(self):
        problem  = self.problem
        self.compute_dist()
        self.time_route     = self.dist / problem.v + problem.t_load*np.sum(self.Y, axis = 3)
        self.time_adding    = self.b / problem.v + problem.t_load
        self.time_substract = self.a / problem.v + problem.t_load

    def compute_inventory(self):
        self.I_s = np.zeros((self.T,self.M))
        self.I_w = np.zeros((self.T,self.N))

        self.I_s[0] = self.problem.I_s_init[:]
        self.I_w[0] = self.problem.I_w_init[:]

        for t in range(1,self.T): 
            self.I_s[t] = self.I_s[t-1]+ self.problem.Q1 * np.sum( self.q[t,:,:,:], axis = (0,1) ) - self.problem.dt[t,:]
            self.I_w[t] = self.I_w[t-1]- self.problem.Q1 * np.sum( self.q[t,:,:,:], axis = (1,2) ) + self.problem.Q2 * self.X[t,:]

        




    def ISI(self, G = 1):
        # change the solution itself to the ISI solution
        problem = self.problem
        T,N,K,M = self.T, self.N, self.K, self.M

        self.compute_a_and_b()
        

        self.compute_time_adding()
        
        # decision variables:
        # q(t,n,k,m): fraction of capacity Q1 of truck k from warehouse m that is delivered to school n at time t
        # delta(t,n,k,m): binary variable, equals 1 if school n is removed from tour performed by truck k from warehouse m at time t, 0 else
        # omega(t,n,k,m): binary variable, equals 1 if school n is inserted into route by truck k from warehouse m at time t, 0 else

        set_q     = [ (t,n,k,m) for t in range(T) for n in range(N) for k in range(K) for m in range(M) if self.Cl[n,m]  ]
        set_delta = [ (t,n,k,m) for (t,n,k,m) in set_q if self.Y[t,n,k,m]  ]
        set_omega = [ (t,n,k,m) for (t,n,k,m) in set_q if not self.Y[t,n,k,m]  ]


        #print("d",problem.d)
        #print("U_s", problem.U_s)
        #print("L_s", problem.L_s)
        #print("U_w", problem.U_w)
        #print("L_w", problem.L_w)
        #print("Is_init",problem.I_s_init)
        #print("Iw_init", problem.I_w_init)


        ISI_model=plp.LpProblem("ISI_Model",plp.LpMinimize)

        # build dictionaries of decision variables:
        q_vars = plp.LpVariable.dicts("q",set_q, cat='Continuous', lowBound=0., upBound=1.)
        X_vars = plp.LpVariable.dicts("X",[(t,n) for t in range(T) for n in range(N)], cat='Binary')
        delta_vars = plp.LpVariable.dicts("delta",set_delta, cat='Binary')
        omega_vars = plp.LpVariable.dicts("omega",set_omega, cat='Binary')
        # just to remember : the psi of the paper is the same thing as our Y
        

        I_s = {(0,m): problem.I_s_init[m]   for m in range(M) }   # need to see how to change an LpAffineExpression with a constant value

        I_w = {(0,n): problem.I_w_init[n]   for n in range(N) }  # need to see how to change an LpAffineExpression with a constant value

        for t in range (1,T): 
            I_s.update(  {(t,m):
                         I_s[t-1,m]
                         + problem.Q1 * plp.lpSum(q_vars[t,n,k,m] for k in range(K) for n in range(N) if self.Cl[n,m] ) 
                         - problem.dt[t,m]
                         for m in range(M) }  
                        )
        
            I_w.update(  {(t,n):
                         I_w[t-1,n]  
                         - problem.Q1 * plp.lpSum(q_vars[t,n,k,m] for k in range(K) for m in range(M) if self.Cl[n,m] ) 
                         + problem.Q2 * X_vars[t,n]
                         for n in range(N) }  
                        )

        transport_cost = problem.c_per_km * plp.lpSum( self.b[t,n,k,m] * omega_vars[t,n,k,m] for (t,n,k,m) in set_omega ) - problem.c_per_km * plp.lpSum( self.a[t,n,k,m] * delta_vars[t,n,k,m] for (t,n,k,m) in set_delta )
        add_cost = plp.lpSum([problem.h_s[m] * I_s[t,m] for t in range(T) for m in range(M)]) + problem.c_per_km * plp.lpSum( problem.to_central[n] * X_vars[t,n] for t in range(T) for n in range(N) ) 

        #objective function
        ISI_model += add_cost + transport_cost, 'Z'
        
        # constraint 9 in Latex script, respect capacities + min. stock of schools and warehouses
        for t in range(1,T):
            # schools: problem.L_s < I_s < problem.U_s
            for m in range(M):
                ISI_model += I_s[t,m] <= problem.U_s[m]       #I_s < U_s
                ISI_model += I_s[t,m] >= problem.L_s[m]      #I_s > L_s
            # warehouses: problem.L_w <I_w < problem.U_w
            for n in range(N):
                ISI_model += I_w[t,n] <= problem.U_w[n]       #I_w < U_w      # can maybe be omitted 
                ISI_model += I_w[t,n] >= problem.L_w[n]      #I_w > L_w

        # constraint on capacity of trucks
        for t in range(T):
            for n in range(N):
                for k in range(K):
                    ISI_model += plp.lpSum(q_vars[t,n,k,m] for  m in range(M)) <=1

        
        # constraint 11: only positive amount to deliver if school is served in that round
        #q < (self.Y - delta + omega)   no need to multiply by U because the component of q is already smaller than 1 because it is normalized by Q1
        for (t,n,k,m) in set_q:     
            if (t,n,k,m) in set_delta :
                ISI_model += q_vars[t,n,k,m] <= 1 - delta_vars[t,n,k,m]
            else : 
                ISI_model += q_vars[t,n,k,m] <= omega_vars[t,n,k,m]

        #constraint 18: bound on the number of changes comitted by the ISI model
        #sum(delta+omega, axis = 3) < G
        for t in range(T):
            for k in range(K):
                ISI_model += plp.lpSum(delta_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_delta ) + plp.lpSum(omega_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_omega ) <= G


        # Constraint on the time spending in one tour
        #sum( omega*self.time_adding, axis  = 3 ) + self.time_route - sum( delta*self.time_substracting, axis  = 3 )  < Tmax
        for t in range(T):
            for n in range(N):
                for k in range(K):
                    expression = self.time_route[t,n,k]
                    expression = expression + plp.lpSum(omega_vars[t,n,k,m] * self.time_adding[t,n,k,m] for m in range(M) if (t,n,k,m) in set_omega )
                    expression = expression - plp.lpSum(delta_vars[t,n,k,m] * self.time_substract[t,n,k,m] for m in range(M) if (t,n,k,m) in set_delta)

                    ISI_model += expression <= problem.Tmax


        ISI_model.solve(solver = plp.GLPK_CMD())

    
        
        #ISI_model.solve()

        # transform the _vars things into numpy array to return it. 

        for (t,n,k,m) in set_q : 
            self.q[t,n,k,m] = q_vars[t,n,k,m].varValue
            if (t,n,k,m) in set_delta : 
                self.Y[t,n,k,m] -= delta_vars[t,n,k,m].varValue
            else : 
                self.Y[t,n,k,m] += omega_vars[t,n,k,m].varValue


        for t in range(T):
            for n in range(N):
                self.X[t,n]=X_vars[t,n].varValue

        add = add_cost.value()
    
        self.compute_r()
        self.compute_costs(add=add)


    def visualization(self,filename):
        km = np.sum(self.dist, axis = (1,2))
        self.compute_inventory()
        visual = visu(self.problem,"WFP Inventory problem", self.I_s,self.I_w, km, self.r, self.X)
        fig = go.Figure(visual)
        offline.plot(fig, filename= filename, auto_open = False)
        

    def __repr__(self):
        km = np.sum(self.dist, axis = (1,2))
        self.compute_inventory()
        visual = visu(self.problem,"WFP Inventory problem", self.I_s,self.I_w, km, self.r)
        fig = go.Figure(visual)
        file = "visu.html"
        offline.plot(fig, filename= file, auto_open = False)
        fig.show()
        return("Solution in file {}  with a total cost of {} ".format(file,round(self.cost)))



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



from operators import operators
class Matheuristic : 
    def __init__(self, initial_solution):

        self.operators = [ {'weight' : 1, 'score': 0 , 'number_used':0, 'function':function, 'name':name } for (name, function) in operators ]

        self.solution = initial_solution
        self.solution_best = initial_solution.copy()
        self.solution_prime = initial_solution.copy()




    def final_algo(self, param):
        # here one can do the final matheuristic described in the paper
        M,N,K,T = self.solution.M, self.solution.N, self.solution.K, self.solution.T
        
        # initialization (step 2 and 3 of the pseudo code)
        self.solution.ISI(G = N)
        self.solution_best = self.solution.copy()

        # line 4 of pseudocode
        epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1], seed = param.seed  )

        tau = param.tau_start
        iterations = 0
        while tau > param.tau_end and iterations < 1000 : 
            
            # line 6 of pseudocode
            i = Matheuristic.choose_operator(self.operators)
            operator = self.operators[i]['function']
            self.solution_prime = self.solution.copy()
            operator(self.solution_prime, param.rho)
            G = N
            self.solution_prime.ISI(G=G)

            if self.solution_prime.cost < self.solution.cost : # line 7
                self.solution = self.solution_prime.copy() # line 8
                G = max(G-1,1)                                  # line 9

                keep_going, i  = True, 0
                while keep_going and i < 1000: 
                    i+=1                              #line 10
                    self.solution_prime.ISI(G=G)  
                    if self.solution_prime.cost < (1+epsilon)*self.solution.cost: 
                        if self.solution_prime.cost < self.solution.cost :              # line 11
                            self.solution = self.solution_prime.copy()              # line 12
                            G = max(G-1,1)                                              # line 13
                        else : G = max(int(param.ksi*(N+M)),1)                          # line 14-15
                                            

                    elif self.solution.cost < self.solution_best.cost :   # line 17
                        self.solution_best = self.solution.copy()    # line 18
                        self.operators[i]['score'] += param.sigmas[0]   # line 19
                        G = max(G-1,1)                                  # line 20
                    else :                                              # line 21
                        self.operators[i]['score'] += param.sigmas[1]   # line 22

                        # little deviation from the pseudocode... we think it is s and not s''
                        if self.solution.cost < (1+epsilon)*self.solution_best :  # line 23
                            G = max(int(param.ksi*(N+M)),1)                       # line 24
                        else : keep_going = False

            elif self.solution_prime.cost < self.solution.cost - np.log(rd.random())*tau: # line 27 # choose theta everytime as a new random value or is it a fixed random value?
                self.solution = self.solution_prime.copy()                         # line 28
                self.operators[i]['score'] += param.sigmas[2]                             # line 29
            
            if iterations % param.delta == 0 :
                epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1], seed = param.seed  )
                # implement update_weights or is this already done?
                self.update_weights(param.reaction_factor)
                self.solution = self.solution_best.copy()
            iterations += 1
            tau = tau*param.cooling

        

                
















# test !

def random_problem(T,N,K,M, seed = None):
    

    np.random.seed(seed)
    Schools = []
    Warehouses = []
    central = np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])
    for i in range(M):
        comsumption =  np.random.randint(low = 1, high = 5)
        lower = comsumption
        capacity = lower + np.random.randint(low = 1, high = 10)
        initial = capacity
        storage_cost = np.random.randint(low = 1, high = 5)
        location = np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])
        Schools.append({'capacity': capacity, 'lower': lower, 'consumption':comsumption,'storage_cost': 0 , 'initial': initial,  'name' : 'School {}'.format(i+1), 'location': location})


    for i in range(N):
        lower = np.random.randint(low = 5, high = 30)
        capacity = lower + np.random.randint(low = 5, high = 50)
        initial  =  capacity
        fixed_cost = np.random.randint(low = 1, high = 10)
        location = np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])

        Warehouses.append({'capacity': capacity, 'lower': lower , 'dist_central': np.linalg.norm(location-central) , 'fixed_cost':  0, 'initial': initial,  'name' : 'Warehouse {}'.format(i+1), 'location': location })

    Q1 = np.random.randint(low = 5, high = 20)*10
    Q2 = np.random.randint(low = 10, high = 30)


    problem = Problem(Schools = Schools, Warehouses = Warehouses,T = T,K = K, Q1 = Q1, Q2 = Q2, v = 40, t_load = 0.5, c_per_km = 0.1, Tmax = 6, central = central)
    
    return problem


