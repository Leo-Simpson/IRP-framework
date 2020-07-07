import numpy as np
import numpy.random as rd
import random
from copy import deepcopy
from OR_tools_solve_tsp import tsp_tour
import pulp as plp 
from scipy.spatial import distance_matrix
import pandas as pd
from time import time
from math import *  # for ceil

import plotly.graph_objects as go 
from plotly import offline
from visu import visu 


class Problem :
    #this is the class that contains the data of the problem
    def __init__(self,Warehouses,Schools,T,K, Q1, Q2, v, t_load, c_per_km, Tmax, central = None, D = None, H= None):
        
        inf = 10000
        
        if type(central) is dict and 'location' in central.keys() and type(central['location']) is np.ndarray: 
            self.central = central['location']  
            Warehouses[0]['capacity'] = inf
            Warehouses[0]['lower'] = -inf
            Warehouses[0]['intial'] = 0
            Warehouses[0]['fixed_cost'] = 0
            self.Warehouses = Warehouses
        elif type(central) is np.ndarray : 
            self.central = central
            central_w = {"capacity": inf, "lower":-inf, "dist_central":0, "fixed_cost":0, "initial": 0, "name": "CENTRAL" , "location": self.central}
            self.Warehouses = [central_w] + Warehouses # list of dictionary {'capacity': ..., 'lower':..., 'fixed_cost': ... , 'initial': ...,  'name' : ..., 'location': ...}
        else : 
            self.central = np.zeros(2)
            central_w = {"capacity": inf, "lower":-inf, "dist_central":0, "fixed_cost":0, "initial": 0, "name": "CENTRAL" , "location": self.central}
            self.Warehouses = [central_w] + Warehouses # list of dictionary {'capacity': ..., 'lower':..., 'fixed_cost': ... , 'initial': ...,  'name' : ..., 'location': ...}



        self.Schools = Schools  # list of dictionary {'capacity': ..., 'lower':..., 'consumption': ...,'storage_cost': ... , 'initial': ...,  'name' : ..., 'location':...}
        self.T = T # time horizon
        if H is None : self.H = T
        else : self.H = H
        self.K = K # number of vehicles
        self.Q1 = Q1 # capacity of the trucks for school deliveries
        self.Q2 = Q2 # capacity of the trucks for warehouses deliveries
        self.v = v # average speed of trucks in km/h
        self.t_load = t_load # average loading/unloading time at schools in hours
        self.c_per_km = c_per_km # average routing cost per kilometer
        self.Tmax = Tmax

        if D is None : 
            locations = [w['location'] for w in self.Warehouses] + [s['location'] for s in self.Schools] 
            self.D = distance_matrix(locations,locations)
        else : 
            self.D = D # distance matrix. Numpy array , NOT pandas dataframe

        for i,w in enumerate(self.Warehouses) : 
            w["dist_central"] = self.D[0,i]

        

    def define_arrays(self):
        self.I_s_init  =  np.array([s["initial"] for s in self.Schools])                 # initial inventory of school
        self.U_s       =  np.array([s["capacity"] for s in self.Schools])                # capactiy upper bound school
        self.L_s       =  np.array([s["lower"]+s["consumption"] for s in self.Schools])  # capacity lower bound school     
        self.h_s       =  np.array([s["storage_cost"] for s in self.Schools])            # storage cost school
        

        self.I_w_init  =  np.array([w["initial"] for w in self.Warehouses])              # initial inventory of warehouses
        self.U_w       =  np.array([w["capacity"] for w in self.Warehouses])             # capactiy upper bound warehouse
        self.L_w       =  np.array([w["lower"] for w in self.Warehouses])                # capacity lower bound warehouse
        self.F_w       =  np.array([w["fixed_cost"] for w in self.Warehouses])           # fixed costs for each warehouse
        self.to_central=  np.array([w["dist_central"] for w in self.Warehouses])         # distance between the warehouses and the central


    def copy(self):
        problem = Problem(self.Warehouses.copy(),Schools.copy(),self.T,self.K,self.Q1,self.Q2,self.v,self.t_load, self.c_per_km,self.Tmax,self.central,self.D, H=self.H)
        problem.define_arrays()
        return problem

    def time_fuse(self,time_step):
        # first copy the problem
        problem = self.copy()

        # then change time horizon : 
        problem.T = ceil(problem.T/time_step)

        # then change the consumption and the prices 
        for s in problem.Schools : 
            s['consumption']  =  s['consumption'] * time_step
            s['storage_cost'] = s['storage_cost'] * time_step

        # then redifine the arrays 
        problem.define_arrays()

        return problem


    def time_defuse(self,time_step) : 
        # first copy the problem
        problem = self.copy()

        # then change time horizon : 
        problem.T = ceil(problem.T*time_step)

        # then change the consumption and the prices 
        for s in problem.Schools : 
            s['consumption']  =  s['consumption'] / time_step
            s['storage_cost'] = s['storage_cost'] / time_step

        # then redifine the arrays 
        problem.define_arrays()

        return problem

    
    def clustering(self):
        problems = []
        # CHRIS : to do 
        return problems 

    






class Solution : 
    #radfactor: parameter to build Cl, gives maximal factor of min distance to warehouse still allowed to serve school, between 1.1 and 1.6
    def __init__(self,problem, Y = None, q = None, X = None, Cl=None, radfactor=100):
        M,N,K,T = len(problem.Schools), len(problem.Warehouses),problem.K, problem.T
        self.M, self.N, self.K, self.T = M,N,K,T

        self.name_schools    = [s["name"] for s in problem.Schools ]
        self.name_warehouses = [w["name"] for w in problem.Warehouses ]
        
        problem.define_arrays()
        self.problem = problem
           
        if Y is None : self.Y = np.zeros((T+1,N,K,M), dtype = bool)      # variable  equal 1 if vehicle k delivers school m from warehouse n at time t
        else         : self.Y = Y
        if q is None : self.q = np.zeros((T+1,N,K,M), dtype = float)     # quantity of food delivered from each warehouse n by vehicle k delivers school m at time t
        else         : self.q = q

        if X is None : self.X = np.zeros((T+1,N), dtype = bool )   # variable equal 1 if warehouse n get more food at time t
        else         : self.X = X

        if Cl is None : self.Cl = np.ones((N,M),dtype=bool)    # equal one when we consider that it is possible that the school m could be served by n
        else          : self.Cl = Cl   
    
        self.r = [[[[] for k in range(self.K)] for n in range(self.N)] for t in range(self.T+1)]
        self.running_time = dict()
        self.feasibility = dict()
        self.feasible = False
        self.a = np.zeros((self.T+1,self.N,self.K,self.M)) # routing cost reduction if school m is removed from the tour of vehicle k from WH n at time t ==> array TxKxMxN
        self.b = np.zeros((self.T+1,self.N,self.K,self.M)) # routing cost addition if school m is added to the tour of vehicle k from WH n at time t ==> array TxKxMxN
        
        d        =  np.array([s["consumption"] for s in problem.Schools])       # consumption of schools
        self.dt  =  np.array([d]*(self.T+1))

        self.build_Cl(radfactor)
        self.compute_costs()


    def copy(self):
        solution = Solution(self.problem,
                             Y = np.copy(self.Y),
                             q = np.copy(self.q),
                             X = np.copy(self.X),
                             Cl= self.Cl      )

        
        solution.r = deepcopy(self.r)
        solution.dt = np.copy(self.dt)
        solution.a = np.copy(self.a)
        solution.b = np.copy(self.b)
        solution.dist = np.copy(self.dist)
        solution.cost = self.cost
        solution.running_time = self.running_time.copy()
        solution.feasibility = self.feasibility.copy()
        solution.feasible = self.feasible

        return solution

        
    def build_Cl(self,radfactor):
        for m in range(self.M):
            dist_vect = self.problem.D[m+self.N][:self.N]

            dist_time = (self.problem.Tmax - self.problem.t_load) * self.problem.v /2          #only warehouses allowed to serve schools that are reachable in tour within Tmax    
            dist_radius = radfactor*np.min(dist_vect)      
                                                     #additionally: only warehouses allowed that are not more than radfactor far away as the closest warehouses
            dist_max = min(dist_time,dist_radius)
            self.Cl[dist_vect > dist_max , m] = False

    
    def compute_school_remove_dist(self,t,n,k):
        tour_complete = [n]+self.r[t][n][k]+[n]
        for i in range(1,len(tour_complete)-1): 
            self.a[t,n,k, tour_complete[i] - self.N] = self.problem.D[tour_complete[i], tour_complete[i+1]] + self.problem.D[tour_complete[i], tour_complete[i-1]] - self.problem.D[tour_complete[i-1], tour_complete[i+1]]
            
    def compute_school_insert_dist(self,t,n,k):
        
        tour_complete   = [n]+self.r[t][n][k]+[n] 

        edges_cost = np.array( [self.problem.D[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)] )
        allowed  = [m for m in range(self.M) if self.Cl[n,m] and not self.Y[t,n,k,m] ]
        #allowed = [m for m in np.where(np.sum(self.Y[t,:,:,:], axis = (0,1)) == 0)[0] if self.Cl[n,m] == 1]
        for m in allowed:
            add_edges_cost =  np.array( [ self.problem.D[m+self.N,tour_complete[i]]+self.problem.D[m+self.N,tour_complete[i+1]]  for i in range(len(tour_complete)-1) ] )
            self.b[t,n,k,m] = np.amin(add_edges_cost-edges_cost)
            
    def cheapest_school_insert(self,t,n,k,m):
        
        tour_school = self.r[t][n][k]
        tour_complete   = [n]+tour_school+[n] 
        edges_cost = np.array( [self.problem.D[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)] )
        add_edges_cost =  np.array( [ self.problem.D[m+self.N,tour_complete[i]]+self.problem.D[m+self.N,tour_complete[i+1]]  for i in range(len(tour_complete)-1) ] )
        position = np.argmin(add_edges_cost)
        cost = add_edges_cost[position]
        return tour_school[:position] + [m] + tour_school[position:], cost
    
    def compute_a_and_b(self):
       
        self.a[:,:,:,:] = 0.
        self.b[:,:,:,:] = 0.

        for t in range(self.T+1): 
            for n in range(self.N):
                for k in range(self.K):

                    self.compute_school_remove_dist(t,n,k)
                        
                    self.compute_school_insert_dist(t,n,k)
                   
    def compute_r(self):
        # here are the TSP to be computed
        self.r = [[[[] for k in range(self.K)] for n in range(self.N)] for t in range(self.T+1)] # for each time t, for each vehicle k, for each warehouse n, a list of ordered integer of [0, M+N] corresponding to a tour
        for t in range(self.T+1):
            for n in range(self.N):
                for k in range(self.K):
                    tour_school = np.nonzero(self.Y[t,n,k,:])[0] + self.N 
                    #tour = [n] + np.ndarray.tolist(np.array(s[0])+self.N) + [n] #  the tour starts at the warehouse then add the school in the wrong order
                    self.r[t][n][k] = tsp_tour(tour_school, n, self.problem.D)  #function returns optimal tour and length of optimal tour
                    # tour without the warehouses, but indexed from N to N+M

    def compute_dist(self):
        self.dist = np.array([[[
                        self.compute_route_dist(self.r[t][n][k],n) for k in range(self.K)
                        ] for n in range(self.N)
                        ] for t in range(self.T+1)
                    ])

    def compute_costs(self): 
        self.compute_dist()
        self.compute_inventory()

        add = np.sum([self.problem.h_s[m] * self.I_s[t,m] for t in range(1,self.T+1) for m in range(self.M)]) + self.problem.c_per_km * np.sum( self.problem.to_central[n] * self.X[t,n] for t in range(1,self.T) for n in range(self.N) ) * 2

        self.cost = self.problem.c_per_km * np.sum(self.dist) + add
        self.cost = self.cost

    def compute_route_dist(self, tour_schools, warehouse : int):
        tour_complete   = [warehouse]+tour_schools+[warehouse]
        distance = sum( [ self.problem.D[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)])
        #distance = distance + something  CHRIS not for wednesday
        return distance

    def compute_time_adding(self):
        problem  = self.problem
        self.compute_dist()
        self.time_route     = self.dist / problem.v + problem.t_load*np.sum(self.Y, axis = 3)
        self.time_adding    = self.b / problem.v + problem.t_load    # to change : t_load can depends on the schools ! 
        self.time_substract = self.a / problem.v + problem.t_load

    def compute_inventory(self):
        self.I_s = np.zeros((self.T+1,self.M))
        self.I_w = np.zeros((self.T+1,self.N))

        self.I_s[0] = self.problem.I_s_init[:]
        self.I_w[0] = self.problem.I_w_init[:]

        for t in range(1,self.T+1): 
            self.I_s[t] = self.I_s[t-1]+ self.problem.Q1 * np.sum( self.q[t,:,:,:], axis = (0,1) ) - self.dt[t,:]
            self.I_w[t] = self.I_w[t-1]- self.problem.Q1 * np.sum( self.q[t,:,:,:], axis = (1,2) ) + self.problem.Q2 * self.X[t,:]

    def verify_feasibility(self):
        self.compute_inventory()
        self.compute_time_adding()
        tol = 1e-4
        self.feasibility = {
                "Truck constraint" : np.all(np.sum(self.q , axis = 3) <= 1 + tol) and np.all(self.q >=-tol),
                "Duration constraint" : np.all(self.time_route <=self.problem.Tmax+ tol ),
                "I_s constraint" : np.all( [ np.all(self.I_s[t]<= self.problem.U_s + tol) and np.all(self.I_s[t]>= self.problem.L_s - tol) for t in range(self.T)]),
                "I_w constraint" : np.all( [ np.all(self.I_w[t]<= self.problem.U_w + tol) and np.all(self.I_w[t]>= self.problem.L_w - tol) for t in range(self.T)])
        }
        self.feasible = self.feasibility["Truck constraint"] and self.feasibility["I_s constraint"] and self.feasibility["I_s constraint"] and self.feasibility["I_w constraint"]
                    


    def ISI(self, G = 1, penalization=10,accuracy = 0.01, time_lim = 1000, solver = "CBC", plot = False, info = True, total_running_time=None):
        # change the solution itself to the ISI solution
        t0 = time()

        problem = self.problem
        T,N,K,M = self.T, self.N, self.K, self.M

        self.compute_a_and_b()
        self.compute_time_adding()
        vehicle_used =  np.any(self.Y, axis=3)
        
        # decision variables:
        # q(t,n,k,m): fraction of capacity Q1 of truck k from warehouse m that is delivered to school n at time t
        # delta(t,n,k,m): binary variable, equals 1 if school n is removed from tour performed by truck k from warehouse m at time t, 0 else
        # omega(t,n,k,m): binary variable, equals 1 if school n is inserted into route by truck k from warehouse m at time t, 0 else

        set_q     = [ (t,n,k,m) for t in range(1,T+1) for n in range(N) for k in range(K) for m in range(M) if self.Cl[n,m]  ]
        set_delta = [ (t,n,k,m) for (t,n,k,m) in set_q if self.Y[t,n,k,m]  ]
        set_omega = [ (t,n,k,m) for (t,n,k,m) in set_q if not self.Y[t,n,k,m] ]


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
        X_vars = plp.LpVariable.dicts("X",[(t,n) for t in range(1,T+1) for n in range(N)], cat='Binary')
        delta_vars = plp.LpVariable.dicts("delta",set_delta, cat='Binary')
        omega_vars = plp.LpVariable.dicts("omega",set_omega, cat='Binary')
        violation_vars = plp.LpVariable.dicts("V",[(t,n,k) for t in range(1,T+1) for n in range(N) for k in range(K)], cat='Binary')
        # just to remember : the psi of the paper is the same thing as our Y

        for (t,n,k,m) in set_q : 
            q_vars[t,n,k,m].setInitialValue(self.q[t,n,k,m])
            if (t,n,k,m) in set_omega : omega_vars[t,n,k,m].setInitialValue(0.)
            elif (t,n,k,m) in set_delta : delta_vars[t,n,k,m].setInitialValue(0.)

        for t in range(1,T+1):
            for n in range(N):
                X_vars[t,n].setInitialValue(self.X[t,n])

        

        # constraint 11: only positive amount to deliver if school is served in that round
        #q < (self.Y - delta + omega)   no need to multiply by U because the component of q is already smaller than 1 because it is normalized by Q1
        for (t,n,k,m) in set_q:     
            if (t,n,k,m) in set_delta :
                ISI_model += q_vars[t,n,k,m] <= 1 - delta_vars[t,n,k,m]
            elif (t,n,k,m) in set_omega : 
                ISI_model += q_vars[t,n,k,m] <= omega_vars[t,n,k,m]



        I_s = {(0,m): problem.I_s_init[m]   for m in range(M) }   # need to see how to change an LpAffineExpression with a constant value

        I_w = {(0,n): problem.I_w_init[n]   for n in range(N) }  # need to see how to change an LpAffineExpression with a constant value

        for t in range (1,T+1): 
            I_s.update(  {(t,m):
                         I_s[t-1,m]
                         + problem.Q1 * plp.lpSum(q_vars[t,n,k,m] for k in range(K) for n in range(N) if self.Cl[n,m] ) 
                         - self.dt[t,m]
                         for m in range(M) }  
                        )
            
            I_w.update(  {(t,n):
                         I_w[t-1,n]  
                         - problem.Q1 * plp.lpSum(q_vars[t,n,k,m] for k in range(K) for m in range(M) if self.Cl[n,m] ) 
                         + problem.Q2 * X_vars[t,n]
                         for n in range(N) }  
                        )
            # constraint 9 in Latex script, respect capacities + min. stock of schools and warehouses
            for m in range(M):
                # schools: problem.L_s < I_s < problem.U_s
                ISI_model += I_s[t,m] <= problem.U_s[m]       #I_s < U_s
                ISI_model += I_s[t,m] >= problem.L_s[m]      #I_s > L_s
            
            for n in range(N):
                # warehouses: problem.L_w <I_w < problem.U_w
                ISI_model += I_w[t,n] <= problem.U_w[n]       #I_w < U_w      # can maybe be omitted 
                ISI_model += I_w[t,n] >= problem.L_w[n]      #I_w > L_w


            for n in range(N):                
                for k in range(K):
                    # constraint on capacity of trucks
                    ISI_model += plp.lpSum(q_vars[t,n,k,m] for  m in range(M) if (t,n,k,m) in set_q ) <=1


            for n in range(N):
                for k in range(K):
                    # Constraint on the time spending in one tour
                    #sum( omega*self.time_adding, axis  = 3 ) + self.time_route - sum( delta*self.time_substracting, axis  = 3 )  < Tmax
                    expression = self.time_route[t,n,k]
                    expression = expression + plp.lpSum(omega_vars[t,n,k,m] * self.time_adding[t,n,k,m] for m in range(M) if (t,n,k,m) in set_omega )
                    expression = expression - plp.lpSum(delta_vars[t,n,k,m] * self.time_substract[t,n,k,m] for m in range(M) if (t,n,k,m) in set_delta)
                    
                    ISI_model += expression <= problem.Tmax + violation_vars[t,n,k]* (sum(self.time_adding[t,n,k])+self.time_route[t,n,k] )
                    #ISI_model += expression <= problem.Tmax


            for n in range(N):
                for k in range(K):
                # constaint for asymetry of the problem in vehicles
                    empties = np.where(~vehicle_used[t,n,:])
                    for i in range(len(empties)-1):
                        k1,k2 = empties[i], empties[i+1]
                        ISI_model += plp.lpSum(omega_vars[t,n,k2,m] for m in range(M) if (t,n,k2,m) in set_omega )<= plp.lpSum(omega_vars[t,n,k1,m] for m in range(M) if (t,n,k1,m) in set_omega )


            for k in range(K):
                #constraint 18: bound on the number of changes comitted by the ISI model
                #sum(delta+omega, axis = 3) < G
                ISI_model += plp.lpSum(delta_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_delta ) + plp.lpSum(omega_vars[t,n,k,m] for n in range(N) for m in range(M) if (t,n,k,m) in set_omega ) <= G


        





        transport_cost = problem.c_per_km * plp.lpSum( self.b[t,n,k,m] * omega_vars[t,n,k,m] for (t,n,k,m) in set_omega ) - problem.c_per_km * plp.lpSum( self.a[t,n,k,m] * delta_vars[t,n,k,m] for (t,n,k,m) in set_delta )
        add_cost = plp.lpSum([problem.h_s[m] * I_s[t,m] for t in range(1,T) for m in range(M)]) + problem.c_per_km * plp.lpSum( problem.to_central[n] * X_vars[t,n] for t in range(1,T) for n in range(N) ) * 2

        violation_cost = penalization* plp.lpSum( violation_vars[t,n,k] for t in range(1,T) for n in range(N) for k in range(K)  )
        #objective function


        ISI_model += add_cost + transport_cost + violation_cost, 'Z'


        
        t1 = time()

        if solver == "CBC"    : ISI_model.solve(solver = plp.PULP_CBC_CMD(msg=False, mip_start=True))
        elif solver == "GLPK" : ISI_model.solve(solver = plp.GLPK_CMD(options=['--mipgap', str(accuracy),"--tmlim", str(time_lim)],msg=0))
        else : raise ValueError(str(solver) + " is not a known solver, use CBC or GLPK")

        t2 = time()

        # transform the _vars things into numpy array to return it. 

        for (t,n,k,m) in set_q : 
            self.q[t,n,k,m] = max(q_vars[t,n,k,m].varValue,0.)
            if (t,n,k,m) in set_delta : 
                self.Y[t,n,k,m] -= delta_vars[t,n,k,m].varValue
            elif (t,n,k,m) in set_omega : 
                self.Y[t,n,k,m] += omega_vars[t,n,k,m].varValue


        for t in range(1,T+1):
            for n in range(N):
                self.X[t,n]=X_vars[t,n].varValue


        add = add_cost.value()
        t3 = time()
        self.compute_r()
        self.compute_costs()
        t4 = time()
        self.verify_feasibility()
        t5 = time()

        self.running_time = { "Define problem" : t1-t0 , "Solve problem ":t2-t1 , "Compute TSPs" : t4-t3  , "Visualisation" : t5-t4}


        if plot : self.visualization('visu.html').show()

        if info : print(self.informations())

        if not total_running_time is None : 
            for name,dt in total_running_time.items() : 
                total_running_time[name] = dt + self.running_time[name]


    def multi_ISI(self,G,solver="CBC", plot = False ,info=True,typ_cost=100,total_running_time=None):
        itera = 50
        c = typ_cost**(1/itera)
        penalization = c
        for p in range(itera):
            self.ISI(G, penalization=penalization,solver = solver, plot = plot, info=info, total_running_time=total_running_time)
            if not self.feasible : 
                print(self)
                raise ValueError("Problem looks infeasible")
            elif not self.feasibility["Duration constraint"] : penalization = penalization*c
            else : return 
        raise ValueError("Not enough penalization for duration constraint")


    def ISI_multi_time(self, G,solver="CBC", plot = False ,info=True,typ_cost=100,total_running_time=None):
        solutions = []
        H = self.problem.H
        L = ceil( self.T/H)
        I_w_init = self.problem.I_w_init
        I_s_init = self.problem.I_s_init
        Tmin, Tmax = 0, min(H,self.T)
        for l in range(L) : 
            sol = self.copy()
            sol.time_cut(Tmin,Tmax)
            sol.problem.I_w_init, sol.problem.I_s_init = I_w_init, I_s_init
            sol.multi_ISI(G,solver = solver, plot = plot, info = info, typ_cost=typ_cost, total_running_time= total_running_time)
            solutions.append(sol)
            I_w_init, I_s_init = sol.I_w[-1], sol.I_s[-1]
            Tmin, Tmax = Tmax, min(Tmax+H,self.T)
             
        self.q[1:] = np.concatenate( [sol.q[1:] for sol in solutions], axis=0  )
        self.X[1:] = np.concatenate( [sol.X[1:] for sol in solutions], axis=0  )
        self.Y[1:] = np.concatenate( [sol.Y[1:] for sol in solutions], axis=0  )
        self.r[1:] = sum( [sol.r[1:] for sol in solutions], [])
        self.compute_costs()
        
        

    def time_cut(self,Tmin,Tmax):
        self.T = Tmax - Tmin 
        self.dt = self.dt[Tmin:Tmax+1]

        self.q = self.q[Tmin:Tmax+1]
        self.X = self.X[Tmin:Tmax+1]
        self.Y = self.Y[Tmin:Tmax+1]
        self.r = self.r[Tmin:Tmax+1]
        
    
    
    
    def visualization(self,filename):
        t0 = time()
        km = np.sum(self.dist, axis = (1,2))
        visual = visu(self.problem,"WFP Inventory problem", self.I_s,self.I_w, km, self.r, self.X, self.q*self.problem.Q1,self.problem.Q2)
        fig = go.Figure(visual)
        offline.plot(fig, filename= filename, auto_open = False)
        self.running_time["visualisation"] = time()-t0
        return fig
        

    def informations(self):
        string_running_time = "Running time : \n  "
        string_f = "Constraints furfilled : \n  "
        for name, t in self.running_time.items():
            string_running_time += name +"  :  " + str(round(t,4)) + "\n  "

        for name, boole in self.feasibility.items():
            string_f += name +"  :  " + str(boole) + "\n  "

        return("Solution with a total cost of {} ".format(round(self.cost),3)
                + " \n "  + string_f
                + "\n "+ string_running_time)


    def __repr__(self):
        self.visualization('visu.html').show()
        return self.informations()
        



class Meta_param : 
    def __init__(self,seed=1):
        self.seed = seed
        rd.seed(self.seed)
        self.Delta = 10
        self.epsilon_bound = (0.05,0.15)
        self.tau_start = 3.
        self.tau_end = 1e-1
        self.cooling = 0.95
        self.reaction_factor = 0.8
        self.sigmas = (10,5,2)
        self.ksi = rd.uniform(low=0.1,high=0.2)
        self.rho_percent = 0.5
        self.max_loop = 100
        self.solver = "CBC"


from operators import operators
class Matheuristic : 
    def __init__(self, problem, seed=1):

        self.operators = [ {'weight' : 1, 'score': 0 , 'number_used':0, 'function':function, 'name':name } for (name, function) in operators ]

        self.solution = Solution(problem)
        #self.solution_best = self.solution.copy()
        #self.solution_prime = self.solution.copy()

        

        self.param = Meta_param(seed=seed)

        




    def algo1(self, param, MAXiter = 1000, solver= "CBC"):
        # here one can do the final matheuristic described in the paper : page 18
        t0 = time()
        rd.seed(param.seed)

        M,N,K,T = self.solution.M, self.solution.N, self.solution.K, self.solution.T
        
        # initialization (step 2 and 3 of the pseudo code)
        self.solution.ISI(G = N, solver=solver)
        running_time = self.solution.running_time.copy()

        self.solution_best = self.solution.copy()

        # line 4 of pseudocode
        epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1]  )

        tau = param.tau_start
        iterations = 0
        while tau > param.tau_end and iterations < MAXiter : 
            
            # line 6 of pseudocode
            i = Matheuristic.choose_operator(self.operators)
            operator = self.operators[i]['function']
            self.solution_prime = self.solution.copy()
            operator(self.solution_prime, param.rho)
            G = N
            self.solution_prime.ISI(G=G, solver=solver)

            if self.solution_prime.cost < self.solution.cost and self.solution_prime.feasible : # line 7
                self.solution = self.solution_prime.copy() # line 8
                G = max(G-1,1)                                  # line 9

                for j in range(param.max_subloop):
                    self.solution_prime.ISI(G=G, solver=solver)  

                    if self.solution_prime.cost < (1+epsilon)*self.solution.cost and self.solution_prime.feasible: 
                        if self.solution_prime.cost < self.solution.cost :              # line 11
                            self.solution = self.solution_prime.copy()              # line 12
                            G = max(G-1,1)                                              # line 13
                        else : G = max(int(param.ksi*(N+M)),1)                          # line 14-15
                                            

                    elif self.solution.cost < (1+epsilon)*self.solution_best.cost and self.solution.feasible:   # line 17 / 23 (deviation from pseudo code : not s'' but s ! )
                        if self.solution.cost < self.solution_best.cost : #line 17
                            self.solution_best = self.solution.copy()    # line 18
                            self.operators[i]['score'] += param.sigmas[0]   # line 19
                            G = max(G-1,1)                                  # line 20
                        else :                                              # line 21
                            self.operators[i]['score'] += param.sigmas[1]   # line 22
                            G = max(int(param.ksi*(N+M)),1)                 # line 24
                    
                    else : 
                        self.operators[i]['score'] += param.sigmas[1]   # line 22
                        break

            elif self.solution_prime.cost < self.solution.cost - np.log(rd.random())*tau and self.solution_prime.feasible: # line 27 # choose theta everytime as a new random value or is it a fixed random value?
                self.solution = self.solution_prime.copy()                         # line 28
                self.operators[i]['score'] += param.sigmas[2]                             # line 29
            
            if iterations % param.Delta == param.Delta-1 :
                epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1])
                # implement update_weights or is this already done?
                self.update_weights(param.reaction_factor)
                self.solution = self.solution_best.copy()
            iterations += 1
            tau = tau*param.cooling

            print("Step %i is finished !!" %iterations)
            print("Current cost is : ", self.solution_best.cost )
        t1 = time()
        visualization(self.solution_best, "solution.html")
        t2 = time()
        print('Total algorithm time = {} <br> Final visualisation time = {} '.format(round(t1-t0,2),round(t2-t1,2)))

        string_running_time ="Total ISI running times : \n "
        for name, t in self.running_time.items():
            string_running_time += name +"  :  " + str(round(t,4)) + "\n  "



    def info_operators(self):
        print("\n Scores of operators :  " )
        for op in self.operators :
            print("w = ",format(op["weight"], '.2f'), " number used = ", op['number_used'], "score = ",op["score"], " name : ", op["name"])
        print("\n")


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
        

    def algo2(self, info = False, plot = False):
        # modified algo :  we don't do line 20, 23, 24
        t0 = time()
        param = self.param
        param.rho = max(int(param.rho_percent * self.solution.M),1)
        rd.seed(param.seed)

        M,N,K,T,p= self.solution.M, self.solution.N, self.solution.K, self.solution.T, 0
        

        self.solution.ISI_multi_time(G = N, solver=param.solver, info=info, plot=plot) 
        typical_cost = self.solution.cost
        self.running_time = self.solution.running_time.copy()
        self.solution_best = self.solution.copy()


        tau, iterations, epsilon = param.tau_start, 0, rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1]  )
        while tau > param.tau_end and iterations < param.max_loop : 
            t0_loop = time()
            i = Matheuristic.choose_operator(self.operators)
            operator = self.operators[i]['function']
            self.operators[i]['number_used'] += 1
            self.solution_prime = self.solution.copy()
            operator(self.solution_prime, param.rho)
            G = N
            self.solution_prime.ISI_multi_time(G=G, solver=param.solver, info=info,typ_cost=typical_cost, total_running_time=self.running_time, plot=plot)
            

            amelioration, finish = False, False
            while ( self.solution_prime.cost < (1+epsilon)*self.solution.cost ):
                if self.solution_prime.cost < self.solution.cost :              
                    self.solution = self.solution_prime.copy()            
                    G = max(G-1,1) 
                    amelioration,finish = True, False                                         
                else : 
                    G = max(int(param.ksi*N),1)   
                    if finish : break
                    finish = True

                self.solution_prime.ISI_multi_time(G=G, solver=param.solver, info=info,typ_cost=typical_cost,total_running_time=self.running_time, plot=plot)

            
            if self.solution.cost < self.solution_best.cost : 
                self.solution_best = self.solution.copy()  
                self.operators[i]['score'] += param.sigmas[0]   

            if amelioration : self.operators[i]['score'] += param.sigmas[1]   
            
            elif self.solution_prime.cost < self.solution.cost - np.log(rd.random())*tau*typical_cost : # choose theta everytime as a new random value or is it a fixed random value?
                self.solution = self.solution_prime.copy()                        
                self.operators[i]['score'] += param.sigmas[2]                       
                

            if iterations % param.Delta == param.Delta-1 :
                self.info_operators()
                epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1])
                # implement update_weights or is this already done?
                self.update_weights(param.reaction_factor)
                self.solution = self.solution_best.copy()

            iterations += 1
            tau = tau*param.cooling
            dt = time()-t0_loop

            print("Step : ", iterations,"Tau : ",round(tau,2), "Current cost is : ",round(self.solution.cost,1) , "Current best cost is : ", round(self.solution_best.cost,1), "Running time : ",round(dt,2) )
        
        t1 = time()
        self.solution_best.visualization("solution.html").show()
        t2 = time()
        print(" Total algorithm time = {} \n Final visualisation time = {} ".format(round(t1-t0,2),round(t2-t1,2)))

        string_running_time ="Total ISI running times : \n  "
        for name, t in self.running_time.items():
            string_running_time += name +"  :  " + str(round(t,2)) + "\n  "

        print(string_running_time )









# test !

def random_problem(T,N,K,M,H = None, seed = None):
    

    np.random.seed(seed)
    Schools = []
    Warehouses = []
    Q2 = np.inf
    
    for i in range(M):
        comsumption =  np.random.randint(low = 1, high = 5)
        lower = 0.
        capacity = comsumption + np.random.randint(low = 1, high = 10)
        initial = capacity
        storage_cost = np.random.randint(low = 1, high = 5)
        location = np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])
        Schools.append({'capacity': capacity, 'lower': lower, 'consumption':comsumption,'storage_cost': 0 , 'initial': initial,  'name' : 'School {}'.format(i+1), 'location': location})


    for i in range(N):
        lower = np.random.randint(low = 5, high = 30)
        liberty = np.random.randint(low = 5, high = 50) # difference between lower bound and capacity
        Q2 = min(Q2,liberty)
        capacity = lower + liberty
        initial  =  capacity
        fixed_cost = np.random.randint(low = 1, high = 10)
        location = np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])

        Warehouses.append({'capacity': capacity, 'lower': lower, 'fixed_cost':  0, 'initial': initial,  'name' : 'Warehouse {}'.format(i+1), 'location': location })

    Q1 = np.random.randint(low = 5, high = 20)
    Q2 = Q2 - 1

    central = np.array([np.random.randint(low = -100, high = 100),np.random.randint(low = -100, high = 100)])
    problem = Problem(Schools = Schools, Warehouses = Warehouses,T = T,K = K, Q1 = Q1, Q2 = Q2, v = 50, t_load = 0.5, c_per_km = 0.1, Tmax = 100, central = central, H = H)
    
    return problem
