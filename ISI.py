import sys
sys.path.append('../')
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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from copy import deepcopy

import plotly.graph_objects as go 
from plotly import offline
from visu import visu 


class Problem :
    #this is the class that contains the data of the problem
    def __init__(self,Warehouses,Schools,
                 Q1, Q2=20, K = None, V_number = None,
                 T = 1, H=None, t_virt = None, time_step = 1,
                 v=40, t_load=0.5, c_per_km=10., Tmax=10, 
                 central = None, central_name = None, makes=None, D = None):
        
        inf = 10000
        if central_name is None : central_name = "CENTRAL"

        if type(central) is np.ndarray :
            central_w = {"capacity": inf, "lower":-inf, "dist_central":0, "fixed_cost":0, "initial": 0, "name": central_name , "location": central}
            Warehouses = [central_w] + Warehouses 
        
        elif type(central) is dict : Warehouses = [central] + Warehouses 

 
           
        self.central = Warehouses[0]['location']
        Warehouses[0]['capacity'] = inf
        Warehouses[0]['lower'] = -inf
        Warehouses[0]['initial'] = 0
        Warehouses[0]['fixed_cost'] = 0
        self.Warehouses = Warehouses
            

        N = len(self.Warehouses)
        self.Schools = Schools  # list of dictionary {'capacity': ..., 'lower':..., 'consumption': ...,'storage_cost': ... , 'initial': ...,  'name' : ..., 'location':...}
        self.T = T # time horizon
        
        if H is None : self.H = T
        else : self.H = H

        if t_virt is None : self.t_virt = 1
        else : self.t_virt = t_virt
        
        if K is None : 
            if V_number is None : raise ValueError("Please put in K or V_number.")
            else : 
                if type(V_number) is np.ndarray : 
                    if V_number.shape == (N,) : 
                        if V_number.dtype == 'int': 
                            self.V_number = V_number
                            if type(Q1) is np.ndarray : 
                                self.Q1 = Q1
                                self.K = Q1.shape[1]
                                if max(V_number) > self.K : raise ValueError("The second dimension of Q1 is {} whereas K is {} (higher)".format(Q1.shape[1],self.K))
                            elif type(Q1) is int : 
                                self.K = max(V_number)
                                self.Q1 = np.ones((N,self.K),dtype=float)*Q1

                        else : raise ValueError("V_number should be integers not {}".format(V_number.dtype))
                    else : raise ValueError("V_number should have a shape of {}, not {}".format((N,),V_number.shape))
                else : raise ValueError("V_number should be an array")
        
        elif type(K) is int :
            if V_number is None : 
                self.K, self.V_number = K, np.ones(N,dtype=int)*K  
                if type(Q1) is np.ndarray : 
                    self.Q1 = Q1
                    if Q1.shape[1] != self.K : raise ValueError("The second dimension of Q1 is {} whereas K is {} ".format(Q1.shape[1],self.K))
                elif type(Q1) is int : 
                    self.Q1 = np.ones((N,self.K),dtype=float)*Q1

            else : raise ValueError("Please choose between entering K and V_number")
        
        else : raise ValueError("K should be an integer, and not a {}".format(type(K)))
        
        if makes is None : self.makes = np.array([["No name    "]*self.K]*N)
        else :self.makes = makes # name of the vehicles
        
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

        self.time_step = time_step
      
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
        problem = Problem(deepcopy(self.Warehouses.copy()),deepcopy(self.Schools),
                            Q1=self.Q1.copy(),Q2=self.Q2, V_number= self.V_number.copy(),
                            T=self.T, H=self.H, t_virt = self.t_virt, time_step = self.time_step,
                            v=self.v,t_load=self.t_load, c_per_km=self.c_per_km,Tmax=self.Tmax,
                            makes = self.makes, D=self.D)

        problem.define_arrays()
        return problem

    def time_fuse(self,time_step=None):
        if time_step is None : time_step = self.time_step
        # then change time horizon : 
        self.T = ceil(self.T/time_step)
        self.H = ceil(self.H/time_step)
        self.t_virt = ceil(self.t_virt/time_step)

        # then change the consumption and the prices 
        for s in self.Schools : 
            s['consumption']  =  s['consumption'] * time_step
            s['storage_cost'] = s['storage_cost'] * time_step

        # then redifine the arrays 
        self.define_arrays()

    def time_defuse(self,time_step) : 
        # then change time horizon : 
        self.T = ceil(self.T*time_step)
        self.H = ceil(self.H*time_step)
        self.t_virt = ceil(self.t_virt*time_step)


        # then change the consumption and the prices 
        for s in problem.Schools : 
            s['consumption']  =  s['consumption'] / time_step
            s['storage_cost'] =  s['storage_cost'] / time_step

        # then redifine the arrays 
        self.define_arrays()

    def augment_v(self,nbr=1):
        self.V_number = self.V_number*nbr
        self.K = self.K *nbr
        self.Q1 = np.repeat(self.Q1,nbr,axis=1)



    def clustering(self):
        
        central_name = self.Warehouses[0]["name"]
        schools = self.Schools
        X = np.array([s['location'] for s in schools])
        
        N = len(self.Warehouses)-1     #determine best amount of clusters
        best_score = 0
        for k_temp in range(max(2,N-3),N+4):
            gmm_class = GaussianMixture(k_temp,random_state=1)     #compute clustering
            gmm_class.fit(X)
            pred = gmm_class.predict(X)
            score = silhouette_score(X, pred, metric='euclidean')
            if score > best_score:
                labels = pred
                centers = gmm_class.means_
                best_score = score
                k = k_temp
        print('Found {} clusters!'.format(k))
        
        schools_div = [[] for i in range(k)]     #split schools into clusters
        for counter, label in enumerate(labels):
            schools_div[label].append(schools[counter])
            
        wh_div = [[] for i in range(k)]            #assign nearest warehouse(s) to every cluster
        v_num_div = [[] for i in range(k)]
        q1_div = [[] for i in range(k)]
        makes_div = [[] for i in range(k)]
        central_in = [False]*k
        for counter, c in enumerate(centers):
            dist = np.array([np.linalg.norm(c - wh['location']) for wh in self.Warehouses])
            min_dist = np.min(dist)
            wh_near = np.where(dist <= 1.1*min_dist)[0]
            for i in wh_near:
                if i == 0 : central_in[counter]=True
                wh_div[counter].append(self.Warehouses[i])
                v_num_div[counter].append(self.V_number[i])
                q1_div[counter].append(self.Q1[i])
                makes_div[counter].append(self.makes[i])
        
        problems = []
        for i in range(k):
            if central_in[i]:
                central = None
            else :
                central = self.Warehouses[0]["location"]
                v_num_div[i].insert(0,0)     # central has no vehicule if it is not "in"
                q1_div[i].insert(0,self.Q1[0])
                makes_div[i].insert(0,self.makes[0])

    
            problems.append(  Problem(wh_div[i], schools_div[i],
                                    Q1 = np.array(q1_div[i]), Q2 = self.Q2,V_number = np.array(v_num_div[i]),
                                    T=self.T, H = self.H, t_virt = self.t_virt, time_step=self.time_step,
                                    v=self.v, t_load=self.t_load, c_per_km=self.c_per_km, Tmax=self.Tmax, 
                                    makes = np.array(makes_div[i]),central=central,central_name=central_name, D=self.D)   )
                

      
        return problems 

    def final_solver(self, param, time_step=1, plot_cluster = True, info = False, folder="solution", comp_small_cl = False, return_var = False):
        self.time_fuse(time_step)
        problems = self.clustering()
        solutions = []
        if return_var:
            heur = []
        for counter, pr in enumerate(problems):
            if not comp_small_cl or len(pr.Schools) <= 55:
                print('Starting to compute cluster {} of {}! (Belonging to WH {})'.format(counter + 1, len(problems), pr.Warehouses[-1]['name']))
                heuristic = Matheuristic(pr,param=param)
                heuristic.algo2(plot_final=plot_cluster,info=info, file = folder+"/cluster %i.html" % (counter+1) )
                solutions.append(heuristic.solution_best)
                if return_var:
                    heur.append(heuristic)
                print('Cluster {} of {} computed!'.format(counter + 1, len(problems)))
            else:
                print('Skipping cluster {} of {} of size {}! (Belonging to WH {})'.format(counter + 1, len(problems), len(pr.Schools),pr.Warehouses[-1]['name']))

        solution = cluster_fusing(solutions,self)
        solution.file = folder+"/global.html"
        if return_var:
            return heur
        else:
            print(solution)
            self.write_in_excel(solution)
    
    def write_in_excel(self, solution):
        # print('Y:')
        # print(solution.Y)
        # print('q:')
        # print(solution.q)
        # print('X:')
        # print(solution.X)
        # print('r:')
        # print(solution.r)
        # print('I_s:')
        # print(solution.I_s)
        output = {'Timestep': [], 'Warehouse': [], 'Vehicle':[], 'Total quantity': [], 'Route': []}
        for t in range(solution.T+1):
            for w in range(solution.N):
                for k in range(solution.V_number[w]):
                    if any(solution.Y[t,w,k,:]):
                        output['Timestep'].append(t)
                        output['Warehouse'].append(solution.name_warehouses[w])
                        output['Vehicle'].append(self.makes[w,k])
                        output['Total quantity'].append(round(self.Q1[w,k]*sum(solution.q[t,w,k,:]),3))
                        school_index = solution.r[t][w][k]
                        school_route = []
                        for i in school_index: school_route.append(solution.name_schools[i-solution.N]+"({})".format(round(self.Q1[w,k]*solution.q[t,w,k,i-solution.N],3)))
                        output['Route'].append(school_route)
        do=pd.DataFrame(output, columns = ['Timestep', 'Warehouse', 'Vehicle', 'Total quantity', 'Route'])
        do.to_excel(r'../Data/Output.xlsx', index = False)

    def __repr__(self):
        toprint = "Parameters of the problem : \n"

        toprint += "         Number of time steps : {} \n".format(self.T)
        toprint += "         Number of schools : {} \n".format(len(self.Schools))
        toprint += "         Number of warehouses : {} \n".format(len(self.Warehouses))
        toprint+= "          Name of the warehouses : {} \n".format([ w["name"] for w in self.Warehouses  ])

        toprint += "   Parameters specific to transportation : \n"
        toprint += "         Number of vehicles in total : {} \n".format(sum(self.V_number))
        toprint += "         Cost per kilometers done : {} \n".format(self.c_per_km)
        toprint += "         Capacity of trucks for pickups : {} \n".format(self.Q2)
        toprint += "         Capacity of trucks are from  : {}  to {}  \n".format(np.max(self.Q1),np.min(self.Q2))

        toprint += "   Parameters specific to time steps : \n"
        toprint += "         Number of time steps : {} \n".format(self.T)
        toprint += "         Length of a time step (in weeks) : {} \n".format(self.time_step)
        toprint += "         Length of time horizon segment for optimization (H) : {} \n".format(self.H)
        toprint += "         Number of virtual time steps between segments : {} \n".format(self.t_virt)

        toprint += "  Duration constraint : \n"
        toprint += "         Maximum time spending in a tour : {} \n".format(self.Tmax)
        toprint += "         Loading time at each school : {} \n".format(self.t_load)
        toprint += "         Speed of vehicules : {} \n".format(self.v)
        toprint += "         Number of virtual time steps between segments : {} \n".format(self.t_virt)



        
        T,self.T = self.T, 0
        sol = Solution(self)
        sol.visualization().show()
        self.T = T

        return toprint



class Solution : 
    #radfactor: parameter to build Cl, gives maximal factor of min distance to warehouse still allowed to serve school, between 1.1 and 1.6
    def __init__(self,problem, Y = None, q = None, X = None, Cl=None, radfactor=100):
        M,N,K,T, V_number = len(problem.Schools), len(problem.Warehouses),problem.K, problem.T, problem.V_number
        self.M, self.N, self.K, self.T, self.V_number = M,N,K,T, V_number

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
        self.file = 'visu.html'
        
        
    def Cl_shaped_like_Y(self):
        Cl_shape_Y = np.zeros(self.Y.shape)
        for i in range(self.T + 1):
            for j in range(self.N):
                for k in range(self.K):
                    Cl_shape_Y[i,j,k,:] = self.Cl[j,:]
        return Cl_shape_Y
    
    def V_num_array(self, shape_Y = False):
        
        V_num_array = np.array([[i for i in range(self.K)] for j in range(self.N)]) < self.V_number.reshape(self.N,1)
        if shape_Y:
            V_num_shaped_Y = np.zeros(self.Y.shape, dtype = bool)
            for i in range(self.T + 1):
                for j in range(self.N):
                    for k in range(self.K):
                        V_num_shaped_Y[i,j,k,:] = V_num_array[j,k]
            return V_num_shaped_Y
        return V_num_array

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
                for k in range(self.V_number[n]):
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
        

    def compute_route_dist(self, tour_schools, warehouse : int):
        tour_complete   = [warehouse]+tour_schools+[warehouse]
        distance = sum( [ self.problem.D[tour_complete[i],tour_complete[i+1]] for i in range(len(tour_complete)-1)])
        #distance = distance + something  CHRIS not for wednesday
        return distance

    def compute_time_adding(self):
        problem  = self.problem
        self.compute_dist()
        self.time_route     = self.dist / problem.v + problem.t_load*np.sum(self.Y, axis = 3)   # to change : t_load can depends on the schools ! 
        self.time_adding    = self.b / problem.v + problem.t_load    
        self.time_substract = self.a / problem.v + problem.t_load

    def compute_inventory(self):
        self.I_s = np.zeros((self.T+1,self.M))
        self.I_w = np.zeros((self.T+1,self.N))

        self.I_s[0] = self.problem.I_s_init[:]
        self.I_w[0] = self.problem.I_w_init[:]

        for t in range(1,self.T+1): 
            self.I_s[t] = self.I_s[t-1]+  np.sum( self.problem.Q1[:,:,np.newaxis] * self.q[t,:,:,:], axis = (0,1) ) - self.dt[t,:]
            self.I_w[t] = self.I_w[t-1]-  np.sum( self.problem.Q1[:,:,np.newaxis] * self.q[t,:,:,:], axis = (1,2) ) + self.problem.Q2 * self.X[t,:]

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
        #self.feasible = self.feasibility["Truck constraint"] and self.feasibility["I_s constraint"] and self.feasibility["I_s constraint"] and self.feasibility["I_w constraint"]

        self.feasible = np.all( [b for name,b in self.feasibility.items()])
                    


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

        set_q     = [ (t,n,k,m) for t in range(1,T+1) for n in range(N) for k in range(self.V_number[n]) for m in range(M) if self.Cl[n,m] ]
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
                         + plp.lpSum(problem.Q1[n,k]*q_vars[t,n,k,m] for k in range(K) for n in range(N) if (self.Cl[n,m] and k< self.V_number[n]) ) 
                         - self.dt[t,m]
                         for m in range(M) }  
                        )
            
            I_w.update(  {(t,n):
                         I_w[t-1,n]  
                         - plp.lpSum(problem.Q1[n,k]*q_vars[t,n,k,m] for k in range(K) for m in range(M) if (self.Cl[n,m] and k< self.V_number[n]) ) 
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
                    # sum( omega*self.time_adding, axis  = 3 ) + self.time_route - sum( delta*self.time_substracting, axis  = 3 )  < Tmax
                    # expression1 is the upper bound of the tour duration
                    # expression2 is the lower bound of the tour duration
                    expression1 = self.time_route[t,n,k]
                    expression1 = expression1 - plp.lpSum(delta_vars[t,n,k,m] * self.time_substract[t,n,k,m] for m in range(M) if (t,n,k,m) in set_delta)
                    expression2 = expression1 + plp.lpSum(omega_vars[t,n,k,m] for m in range(M) if (t,n,k,m) in set_omega ) * problem.t_load        # to change if t_load depends on the schools
                    expression1 = expression1 + plp.lpSum(omega_vars[t,n,k,m] * self.time_adding[t,n,k,m] for m in range(M) if (t,n,k,m) in set_omega )
                    
                    
                    
                    ISI_model += expression1 <= problem.Tmax + violation_vars[t,n,k]* (sum(self.time_adding[t,n,k])+self.time_route[t,n,k] )
                    ISI_model += expression2 <= problem.Tmax


            for n in range(N):
                for k in range(K):
                # constaint for asymetry of the problem in vehicles : if k1 and k2 are initially empty sum(omega, k = k1) <= sum(omega, k=k2)
                    empties = np.where(~vehicle_used[t,n,:])
                    for i in range(len(empties)-1):
                        k1,k2 = empties[i], empties[i+1]
                        if problem.Q[k1] == problem.Q[k2]:
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


    def multi_ISI(self,G,solver="CBC", plot = False ,info=True,total_running_time=None):
        itera = 5
        for p in range(1,itera+1):
            penalization = 2*self.cost * p / itera
            self.ISI(G, penalization=penalization,solver = solver, plot = plot, info=info, total_running_time=total_running_time)
            
            if not (self.feasibility["Truck constraint"] and self.feasibility["I_s constraint"] and self.feasibility["I_w constraint"]):
                print(self)
                raise ValueError("Problem looks infeasible")
            elif not self.feasibility["Duration constraint"] : 
                continue
            else : 
                break
        
        if not self.feasibility["Duration constraint"] : 
            #raise ValueError("Duration constraint looks infeasible")
            print("Duration constraint looks infeasible")


    def ISI_multi_time(self, G,solver="CBC", plot = False ,info=True,total_running_time=None):
        H = self.problem.H
        t_virt = ceil(self.problem.t_virt)
        L = ceil( self.T/H)
        I_w_init = self.problem.I_w_init
        I_s_init = self.problem.I_s_init
        Tmin, Tmax = 0, H+t_virt

        solutions = []
        for l in range(L) : 
            sol = self.copy()
            sol.time_cut(Tmin,Tmax, self.T)
            sol.problem.I_w_init, sol.problem.I_s_init = I_w_init, I_s_init
            sol.multi_ISI(G,solver = solver, plot = plot, info = info, total_running_time= total_running_time)
            
            I_w_init, I_s_init = sol.I_w[-1-t_virt], sol.I_s[-1-t_virt]
            
            solutions.append(sol)
            
            Tmin, Tmax = Tmax - t_virt, Tmax+H+t_virt


        if t_virt == 0 : 
            self.q[1:] = np.concatenate( [sol.q[1:] for sol in solutions], axis=0  )
            self.X[1:] = np.concatenate( [sol.X[1:] for sol in solutions], axis=0  )
            self.Y[1:] = np.concatenate( [sol.Y[1:] for sol in solutions], axis=0  )
            self.r[1:] = sum(            [sol.r[1:] for sol in solutions], [])
        else :
            self.q[1:] = np.concatenate( [sol.q[1:-t_virt] for sol in solutions]+[solutions[-1].q[-t_virt:]], axis=0  )
            self.X[1:] = np.concatenate( [sol.X[1:-t_virt] for sol in solutions]+[solutions[-1].X[-t_virt:]], axis=0  )
            self.Y[1:] = np.concatenate( [sol.Y[1:-t_virt] for sol in solutions]+[solutions[-1].Y[-t_virt:]], axis=0  )
            self.r[1:] = sum(            [sol.r[1:-t_virt] for sol in solutions]+[solutions[-1].r[-t_virt:]], [])
        self.compute_costs()
        self.verify_feasibility()
        
        

    def time_cut(self,Tmin,Tmax, T_total):
        Tmax2 = min(Tmax, T_total)
        self.T = Tmax2 - Tmin 
        self.dt = self.dt[Tmin:Tmax2+1]

        self.q = self.q[Tmin:Tmax2+1]
        self.X = self.X[Tmin:Tmax2+1]
        self.Y = self.Y[Tmin:Tmax2+1]
        self.r = self.r[Tmin:Tmax2+1]
        
    
    
    
    def visualization(self):
        t0 = time()
        schools,warehouses = self.problem.Schools, self.problem.Warehouses
        km = np.sum(self.dist, axis = (1,2))
        visual = visu(schools,warehouses, "WFP Inventory problem", self.I_s,self.I_w, km, self.r, self.X, self.q*self.problem.Q1[np.newaxis,:,:,np.newaxis],self.problem.Q2, self.problem.D, self.problem.makes)
        fig = go.Figure(visual)
        offline.plot(fig, filename= self.file, auto_open = False)
        self.running_time["visualisation"] = time()-t0
        return fig
        

    def informations(self):
        self.verify_feasibility()
        string_running_time = "Running time : \n  "
        string_f = "Constraints fulfilled : \n  "
        for name, t in self.running_time.items():
            string_running_time += name +"  :  " + str(round(t,4)) + "\n  "

        for name, boole in self.feasibility.items():
            string_f += name +"  :  " + str(boole) + "\n  "

        return("Solution with a total cost of {} ".format(round(self.cost,3))
                + " \n "  + string_f
                + "\n "+ string_running_time)


    def __repr__(self):
        self.visualization().show()
        return self.informations()
        



class Meta_param : 
    def __init__(self,seed=1):
        self.seed = seed
        rd.seed(self.seed)
        self.Delta = 10
        self.epsilon_bound = (0.05,0.15)
        self.tau_start = 3.
        self.tau_end = 1e-1
        self.cooling = 0.9
        self.reaction_factor = 0.8
        self.sigmas = (10,5,2)
        self.ksi = rd.uniform(low=0.1,high=0.2)
        self.rho_percent = 0.3
        self.max_loop = 100
        self.solver = "CBC"


    def __repr__(self):
        toprint = " Meta parameters of the heurestics : \n \n "

        toprint += "For iterations : \n "
        toprint += "    Length of segments (delta) : {} , \n ".format(self.Delta)
        tau_iterations = int(np.log(self.tau_end / self.tau_start) / np.log(self.cooling))
        toprint += "    Tau (the temperature) begins at {}, finishes at {} and is cooled by {} --> {} steps \n ".format(self.tau_start, self.tau_end, self.cooling, tau_iterations)
        
        toprint += " \n "

        toprint += "Inside algorithm's parameters : \n "
        toprint += "    Bounds of epsilon  : {} , \n ".format(self.epsilon_bound)
        toprint += "    For randomizing G (ksi) : {} \n".format(round(self.ksi,3))

        toprint += " \n "

        toprint += "    Operators' parameters : \n "
        toprint += "    Sigmas for updating the weigths of the operators {}, with reaction factor : {} , \n ".format(self.sigmas, self.reaction_factor)
        toprint += "    Percentage rho for operators : {} , \n ".format(self.rho_percent)
        toprint += "    Solver used for the MIP : {} ".format(self.solver)

        return toprint
        



from operators import operators
class Matheuristic : 
    def __init__(self, problem,param=None, seed=1):

        self.operators = [ {'weight' : 1, 'score': 0 , 'number_used':0, 'function':function, 'name':name } for (name, function) in operators ]

        self.solution = Solution(problem)
        #self.solution_best = self.solution.copy()
        #self.solution_prime = self.solution.copy()

        

        if param is None  : self.param = Meta_param(seed=seed)
        else              : self.param = param   

        




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
        visualization(self.solution_best)
        t2 = time()
        print('Total algorithm time = {} <br> Final visualisation time = {} '.format(round(t1-t0,2),round(t2-t1,2)))

        string_running_time ="Total ISI running times : \n "
        for name, t in self.running_time.items():
            string_running_time += name +"  :  " + str(round(t,4)) + "\n  "



    def info_operators(operators):
        print("\n Scores of operators :  " )
        for op in operators :
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
        

    def algo2(self, info = False, plot = False,plot_final = False, file = "solution.html",penal=10):
        # modified algo :  we don't do line 20, 23, 24
        t0 = time()
        param = self.param
        param.rho = max(int(param.rho_percent * self.solution.M),1)
        rd.seed(param.seed)
        self.running_time = { "Define problem" : 0., "Solve problem ":0. , "Compute TSPs" : 0. , "Visualisation" : 0.}

        M,N,K,T,p= self.solution.M, self.solution.N, self.solution.K, self.solution.T, 0
        
        self.solution.ISI_multi_time(G = M, solver=param.solver, info=info, total_running_time=self.running_time, plot=plot)
        typical_cost = self.solution.cost
        self.solution.cost += penal*(1-self.solution.feasible)
        self.solution_best = self.solution.copy()


        tau, iterations, epsilon = param.tau_start, 0, rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1]  )
        
        step = {"Step":0, "Tau":round(tau,2),"Current cost":round(self.solution.cost,1),"Current best cost":round(self.solution_best.cost,1),"Running time" : round(time()-t0,2)}
        #print("Step : ", 0,"Tau : ",round(tau,2), "Current cost is : ",round(self.solution.cost,1) , "Current best cost is : ", round(self.solution_best.cost,1), "Running time : ",round(time()-t0,2) )
        #print(step)
        self.steps = [step]
        self.operators_infos = []
        
        while tau > param.tau_end and iterations < param.max_loop : 
            t0_loop = time()
            i = Matheuristic.choose_operator(self.operators)
            operator = self.operators[i]['function']
            self.operators[i]['number_used'] += 1
            self.solution_prime = self.solution.copy()
            operator(self.solution_prime, param.rho)
            G = M
            self.solution_prime.ISI_multi_time(G=G, solver=param.solver, info=info, total_running_time=self.running_time, plot=plot)
            self.solution_prime.cost += penal*(1-self.solution.feasible)

            amelioration, finish = False, False
            while ( self.solution_prime.cost < (1+epsilon)*self.solution.cost ):
                if self.solution_prime.cost < self.solution.cost :              
                    self.solution = self.solution_prime.copy()            
                    G = max(G-1,1) 
                    amelioration,finish = True, False                                         
                else : 
                    G = max(int(param.ksi*M),1)   
                    if finish : break
                    finish = True

                self.solution_prime.ISI_multi_time(G=G, solver=param.solver, info=info,total_running_time=self.running_time, plot=plot)
                self.solution_prime.cost += penal*(1-self.solution.feasible)

            
            if self.solution.cost < self.solution_best.cost : 
                self.solution_best = self.solution.copy()  
                self.operators[i]['score'] += param.sigmas[0]   

            if amelioration : self.operators[i]['score'] += param.sigmas[1]   
            
            elif self.solution_prime.cost < self.solution.cost - np.log(rd.random())*tau*typical_cost : # choose theta everytime as a new random value or is it a fixed random value?
                self.solution = self.solution_prime.copy()                        
                self.operators[i]['score'] += param.sigmas[2]                       
                

            if iterations % param.Delta == param.Delta-1 :
                #Matheuristic.info_operators(self.operators)
                self.operators_infos.append(deepcopy(self.operators))
                epsilon = rd.uniform (low = param.epsilon_bound[0], high = param.epsilon_bound[1])
                # implement update_weights or is this already done?
                self.update_weights(param.reaction_factor)
                self.solution = self.solution_best.copy()

            iterations += 1
            tau = tau*param.cooling
            dt = time()-t0_loop

            step = {"Step":iterations, "Tau":round(tau,2),"Current cost":round(self.solution.cost,1),"Current best cost":round(self.solution_best.cost,1),"Running time" : round(dt,2)}
            #print("Step : ", iterations,"Tau : ",round(tau,2), "Current cost is : ",round(self.solution.cost,1) , "Current best cost is : ", round(self.solution_best.cost,1), "Running time : ",round(dt,2) )
            #print(step)
            self.steps.append(step)


        t1 = time()
        print(" Total algorithm time = {} ".format(round(t1-t0,2)))
        self.solution_best.file = file
        if plot_final : 
            self.solution_best.visualization().show()
            t2 = time()
            print("Final visualisation time = {} ".format(round(t2-t1,2)))
            
        
        

        string_running_time ="Total ISI running times : \n  "
        for name, t in self.running_time.items():
            string_running_time += name +"  :  " + str(round(t,2)) + "\n  "

        print(string_running_time )
        #print(self.solution_best.informations())




    def print_ope(self):
        for i,operators in enumerate(self.operators_infos) : 
            print("Segment of time steps number %i"%i)
            Matheuristic.info_operators(operators)
            print(" ")

    def print_steps(self):
        for step in self.steps:
            print(step)





# test !

def random_problem(T,N,M,K = None, H = None, seed = None):
    

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

    if K is None: 
        V_number = np.array([np.random.randint(low = 2, high = 6) for i in range(N)])
        Q1 = np.zeros((N, max(V_number)))
        for n in range(N):
            for k in range(max(V_number)):
                if k < V_number[n]: Q1[n,k] = np.random.randint(low = 2, high = 20)
    else : 
        Q1 = np.random.randint(low = 5, high = 20)
        V_number = None
    
    Q2 = Q2 - 1

    problem = Problem(Schools = Schools, Warehouses = Warehouses,T = T,K = K, Q1 = Q1, Q2 = Q2, v = 50, t_load = 0.5, c_per_km = 0.1, Tmax = 100, H = H, V_number = V_number)
    
    return problem


def cluster_fusing(solutions, problem_global):
    if solutions : 
        S_names = [s["name"] for s in problem_global.Schools]
        Wh_names = [w["name"] for w in problem_global.Warehouses]


        solution = Solution(problem_global)

        

        for sol in solutions:
            S_ind = [S_names.index(s["name"]) for s in sol.problem.Schools]
            Wh_ind = [Wh_names.index(w["name"]) for w in sol.problem.Warehouses]

            solution.X[:,Wh_ind] += sol.X
            q = solution.q[:,Wh_ind]
            Y = solution.Y[:,Wh_ind]
            q[:,:,:,S_ind] += sol.q
            Y[:,:,:,S_ind] += sol.Y

            solution.q[:,Wh_ind] = q
            solution.Y[:,Wh_ind] = Y
                

        solution.compute_r()
        solution.compute_costs()      # contains compute_inventory() and compute_dist()
        solution.verify_feasibility() # contains compute_inventory()

        for name,t in solution.running_time.items():
            solution.running_time = sum( sol.running_time[name] for sol in solutions )


        return solution 
    
    else : 
        raise ValueError("List of solutions is empty")




def excel_to_pb(path,nbr_tours=1):
        df_w = pd.read_excel(io=path, sheet_name='Warehouses')         #reads the excel table Warehouses
        warehouses = df_w.to_dict('records')                   #and transforms it into a Panda dataframe
        for w in warehouses:                                   #puts longitude and latitude together in a numpy array 'location'
            location = np.array([w['Latitude'],w['Longitude']])
            del w['Latitude'], w['Longitude']
            w['location']=location
            w['name'] = w.pop('Name')
            w['capacity'] = w.pop('Capacity')
            w['lower'] = w.pop('Lower')
            w['initial'] = w.pop('Initial')
            w['fixed_cost'] = w.pop('Fixed Cost')
            
    
        df_s = pd.read_excel(io=path, sheet_name='Schools')
        schools = df_s.to_dict('records')
        for m,s in enumerate(schools):                                      #puts longitude and latitude together in a numpy array 'location'
            location = np.array([s['Latitude'],s['Longitude']])
            del s['Latitude'], s['Longitude']
            s['location']=location
            s['name'] = s.pop('Name_ID') + ' ' + str(m)
            s['lower'] = s.pop('Lower')
            s['initial'] = s.pop('Initial')
            s['consumption'] = s.pop('Consumption per week in MT')
            s['storage_cost'] = s.pop('Storage Cost')
            s['capacity'] = s.pop('Capacity')
            del s['Total Sum of Beneficiaries']
            del s['Total Sum of Commodities']
            del s['Consumption per day in MT']
            
        df_v = pd.read_excel(io=path, sheet_name='VehicleFleet')
        vehicles = df_v.to_dict('records') # list of dictionaries of the form {'Warehouse':...,'Plate Nr':....,'Make':...,'Model':....,'Capacity in MT':....}
        vehicles.sort(key = lambda v : v['Capacity in MT'],reverse = True)
        

        
        i = 0
        # list with N entries, which contain the list of dictionaries {'Warehouse':...., 'Plate Nr':....., 'Capacity in MT':...} per Warehouse
        # self.vehicle_list[i] gives you the list of vehicles(dictionaries) of warehouse i
        vehicle_list=[ [] for w in warehouses ]
        Wh_names = [w['name'] for w in warehouses]
        

        for v in vehicles:
            i = Wh_names.index(v['Warehouse'])
            del v['Model']
            for j in range(nbr_tours):
                vehicle_list[i].append(v)
                    

        V_number = np.array([len(vehicle_list[j]) for j in range(len(warehouses))])
        Q1 = np.zeros((len(warehouses), max(V_number)))
        makes = np.array([["Doesn't exist        "]*max(V_number)]*len(warehouses))
        for n in range(len(warehouses)):
            for k in range(V_number[n]):
                Q1[n,k] = vehicle_list[n][k]['Capacity in MT']
                makes[n,k] = vehicle_list[n][k]['Make']

        return schools, warehouses, Q1, V_number, makes
        


