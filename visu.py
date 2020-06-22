import plotly.graph_objects as go 
import numpy as np
import pandas as pd



COLORS = {"school" : "green", "warehouse": "blue", "central":"red","road":"grey"}
SIZES = {"school" : 30, "warehouse": 40,"central": 50}
TITLE = 'WFP Inventory problem'



def make_annotations(tuples):
    annotations = []
    for (pos,txt,color) in tuples:
        annotations.append(
            dict(
                text=txt, # text within the circles
                x=pos[0], y=pos[1],
                xref='x1', yref='y1',
                font=dict(color=color, size=15),
                showarrow=False)
        )
    return annotations




def plots(schools, warehouses, central):
    x, y, text = [],[],[]
    for s in schools:
        
        x.append(s["location"][0])
        y.append(s["location"][1])
        
        t = ""
        for param in ["name","capacity","lower", "consumption", "storage_cost"] :
            t += param + " = " + str(s[param]) + "<br>"

        text.append(t)
    
    
    plot_s = go.Scatter(x=x, y=y, mode='markers',
                              name='schools',
                              marker=dict(symbol='circle-dot',
                                                size=SIZES["school"],
                                                color=COLORS["school"]
                                                ),
                              text=text,
                              hoverinfo='text',
                              opacity=0.8
                              )


    x, y, text = [],[],[]
    for w in warehouses:

        x.append(w["location"][0])
        y.append(w["location"][1])
        
        t = ""
        for param in ["name","capacity","lower", "fixed_cost"] :
            t += param + " = " + str(w[param]) + "<br>"

        text.append(t)
    
    
    plot_w = go.Scatter(x=x, y=y, mode='markers',
                              name='warehouses',
                              marker=dict(symbol='circle-dot',
                                                size=SIZES["warehouse"],
                                                color=COLORS["warehouse"]
                                                ),
                              text=text,
                              hoverinfo='text',
                              opacity=0.8
                              )

    plot_c = go.Scatter(x=[central[0]],                      
                                y=[central[1]],
                                mode='markers',
                                name='central',
                                marker=dict(symbol='circle-dot',
                                                size=SIZES["central"],
                                                color=COLORS["central"]
                                                ),
                                text=["Central warehouse"],
                                hoverinfo='text',
                                opacity=0.8
                                )
    return [plot_s, plot_w, plot_c]





def make_layout(title,central, pos_s, pos_w, I_s, I_w): 
    
    tuples = [(pos_s[i], "I = "+str(I_s[i]),'black') for i in range(len(I_s))] 
    tuples = tuples + [(pos_w[j], "I = "+str(I_w[j]),'yellow') for j in range(len(I_w))]
    tuples.append((central,"CENTRAL", 'black') )

    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
    zeroline=True,
    showgrid=True,
    showticklabels=True,
    )
    
    return dict(
                title= title,
                  annotations= make_annotations(tuples), 
                  font_size=12,
                  showlegend=False,
                  xaxis=axis,
                  yaxis=axis,
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(248,248,248)',
                  updatemenus= []
            )


def visu(problem, title, I_s, I_w):
    

    pos_s = [s["location"] for s in problem.Schools]
    pos_w = [w["location"] for w in problem.Warehouses]

    data = plots(problem.Schools, problem.Warehouses, problem.central)
    layout = make_layout(title,problem.central,pos_s,pos_w,I_s,I_w)

    return dict(data=data, layout=layout)




























#
#class School :
#    def __init__(self, position, capacity,consumption,name):
#        self.name = name
#        self.pos = position
#        self.C = capacity
#        self.inventory = capacity
#        self.q = consumption
#
#    def eat(self):
#        self.inventory -= self.q
#        
#    def receive(self,x):
#        self.inventory += x
#        
#class Warehouse : 
#    def __init__(self, position, capacity,name):
#        self.name = name
#        self.pos = position
#        self.C = capacity
#        self.inventory = capacity
#        self.pickup_cost = 0
#
#    def deliver(self, quantity):
#        self.inventory -= quantity
#        
#    def receive(self,quantity):
#        self.inventory += quantity
#        
#    def compute_pickup_cost(self,central):
#        self.pickup_cost = round( 2*np.sqrt( (self.pos[0]-central[0])**2 + (self.pos[1]-central[1])**2) )
#    
#    def make_arrow(self,central):
#        
#        x1 = central[0]
#        y1 = central[1]
#        x2 = self.pos[0]
#        y2 = self.pos[1]
#        
#        text = "cost = " + str(round(self.pickup_cost,2))
#
#        return go.Scatter(x=[(7*x1+x2)/8,(x1+7*x2)/8],
#                y=[(7*y1+y2)/8,(y1+7*y2)/8],
#                line= dict(color=COLORS["road"]),
#                text = [text,text],
#                hoverinfo='text',
#                visible = False
#                    )
#
#class Route : 
#    def __init__(self, warehouse, SCHOOLS):
#        self.w = warehouse
#        self.S = SCHOOLS
#        self.Mk = len(self.S)
#
#    def set_Q(self,Q):
#        self.Q = Q      
#
#    def compute_edges(self):
#        '''
#        Make a list of edges for the route, that is, for each edge, a couple of objects Warehouse or School
#        '''
#        self.edges = [(self.w,self.S[0])]
#        for i in range(self.Mk-1):
#            self.edges.append((self.S[i],self.S[i+1]))
#        self.edges.append((self.S[-1],self.w))
#
#    def cost(edge,D):
#        return D.loc[edge[0].name, edge[1].name]
#
#    def compute_costs(self,D,fixed_costs=0.):
#        '''
#        compute the cost of a tour with respect to the distances and the fixed cost
#        '''
#        self.costs = []
#        for e in self.edges :
#            self.costs.append(Route.cost(e,D))
#        
#        self.cost = sum(self.costs) + fixed_costs
#
#    def compute_C(self):
#        self.room_available = [s.C-s.inventory for s in self.S]
#        self.C = min(self.Q, sum(self.room_available))
#                
#    def do_tour(self):
#        '''
#        Change the inventory with respect to a tour and its X
#        '''
#        self.w.deliver(sum(self.X))
#        for i in range(len(self.X)):
#            self.S[i].receive(self.X[i])
#
#    def make_arrows(self) : 
#        self.arrows= []
#        for e in range(self.Mk+1):
#            edge = self.edges[e]
#            x1, x2, y1, y2 = edge[0].pos[0], edge[1].pos[0], edge[0].pos[1], edge[1].pos[1]
#            text = "cost = "+ str(round(self.costs[e],1))
#
#            arrow = go.Scatter(x=[(7*x1+x2)/8,(x1+7*x2)/8],
#                    y=[(7*y1+y2)/8,(y1+7*y2)/8],
#                    line= dict(color=COLORS["road"]),
#                    text = [text,text],
#                    hoverinfo='text',
#                    visible = False
#                    )
#
#            self.arrows.append(arrow)
#            
#    def set_no(self,indices):
#        names = [s.name for s in self.S]
#        self.numbers_RtoS = np.array([indices[name] for name in names  ])
#
#
#class Map :
#    def __init__(self, central, schools, warehouses, possible_routes = [], Q = 5., D=None, Q2 = 10.):
#        '''
#        This object represents the state of the problem at some time t 
#        '''
#        
#        self.central = central
#        self.S = schools
#        self.W = warehouses
#        self.R_possible = possible_routes
#        self.Q = Q
#        self.Q2= Q2
#        self.D = D
#        self.M, self.N, self.K = len(schools),len(warehouses), len(possible_routes )
#        self.warehouse_to_deliver = []
#        self.compute_pickup_costs()
#        
#        
#        find_index_s =  {schools[i].name : i for i in range(self.M)}
#        for r in self.R_possible : 
#            r.set_Q(Q)
#            r.set_no(find_index_s)
#            
#        if D is None :
#            positions = np.array([w.pos for w in warehouses] + [s.pos for s in schools]) 
#            d = Map.compute_distances(positions)
#            names = [w.name for w in warehouses]+[s.name for s in schools]
#            self.D = pd.DataFrame(data=d, columns=names, index=names)
#            
#        else: 
#            self.D = D
#
#        
#        self.t = 0.
#        self.total_cost = 0
#        self.cost = 0
#        
#        for k in range(self.K): 
#            self.R_possible[k].number = k
#
#    def build_Rmat(self):
#        '''
#        Method that build the numpy array Rmat such that r[k,m] = 1 if & only if S[m] in R[k]
#        '''
#        self.Rmat = np.zeros((self.K,self.M),dtype = bool)
#        names = [s.name for s in self.S]
#        for k in range(self.K):
#            r = self.R_possible[k]
#            for stop in r.S : 
#                m = names.index(stop.name)
#                self.Rmat[k,m] = True
#
#    def compute_pickup_costs(self):
#        for w in self.W :
#            w.compute_pickup_cost(self.central)
#    
#    def compute_distances(positions):
#        return np.sqrt(np.sum((positions-positions[:,np.newaxis])**2,axis=2))
#                            
#    def compute_central_deliveries(self):
#        for j in range(self.N):
#            if self.W[j].inventory < 0 : 
#                self.warehouse_to_deliver.append(j)
#                self.W[j].receive(self.Q2)
#    
#    def compute_edges(self):
#        for r in self.R_possible : 
#            r.compute_edges()
#
#    def compute_costs(self, fixed_costs=0.):
#        for r in self.R_possible : 
#            r.compute_costs(self.D, fixed_costs=fixed_costs)
#
#    def select_tours(self,y):
#        # y should be a boolean vector of length K here that state if the tour k should be done
#        self.R = np.arange(self.K)[y]
#        
#    def compute_X(self,x):
#        # x is a matrix KxM that gives the quantity of food to be delivered to each school for each tour
#        for k in self.R : 
#            r = self.R_possible[k]
#            r.X = x[k,r.numbers_RtoS]
#                        
#    def do_tours(self):
#        for k in self.R :
#            r = self.R_possible[k] 
#            r.do_tour()
#            self.cost += r.cost
#        self.compute_central_deliveries()
#        for j in self.warehouse_to_deliver : 
#            self.cost+=self.W[j].pickup_cost
#        
#        
#        self.total_cost += self.cost
#        self.title = TITLE + "        Cost = %s          Total Cost = " %str(round(self.cost)) + str(round(self.total_cost))
#        self.title = self.title + "   Truck 1 capacity : "+ str(self.Q) + "   Truck 2 capacity : "+ str(self.Q2)
#        
#        self.cost = 0
#
#    def eat(self):
#        for s in self.S:
#            s.eat()
#
#    def init_draw(self):
#
#        # create arrows
#        for k in range(self.K): 
#            r = self.R_possible[k].make_arrows()
#
#        #plot the schools
#        plot_schools = go.Scatter(x=[school.pos[0] for school in self.S],
#                        y=[school.pos[1] for school in self.S],
#                        mode='markers',
#                        name='schools',
#                        marker=dict(symbol='circle-dot',
#                                        size=SIZES["school"],
#                                        color=COLORS["school"]
#                                        ),
#                        text=[s.name+" C = "+str(s.C)+"  ;   q = "+str(s.q) for s in self.S],
#                        hoverinfo='text',
#                        opacity=0.8
#                        )
#        #plot the warehouses
#        plot_warehouses = go.Scatter(x=[warehouse.pos[0] for warehouse in self.W],
#                            y=[warehouse.pos[1] for warehouse in self.W],
#                            mode='markers',
#                            name='warehouses',
#                            marker=dict(symbol='circle-dot',
#                                            size=SIZES["warehouse"],
#                                            color=COLORS["warehouse"]
#                                            ),
#                            text=[w.name+" C = "+str(w.C) for w in self.W],
#                            hoverinfo='text',
#                            opacity=0.8
#                            )
#        
#        plot_central = go.Scatter(x=[self.central[0]],
#                            y=[self.central[1]],
#                            mode='markers',
#                            name='central',
#                            marker=dict(symbol='circle-dot',
#                                            size=SIZES["central"],
#                                            color=COLORS["central"]
#                                            ),
#                            text=["Central warehouse"],
#                            hoverinfo='text',
#                            opacity=0.8
#                            )
#
#        
#        axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
#            zeroline=True,
#            showgrid=True,
#            showticklabels=True,
#            )
#
#        self.title = TITLE + "        Cost = %s          Total Cost = " %str(self.cost) + str(self.total_cost)
#        self.title = self.title + "   Truck 1 capacity : "+ str(self.Q) + "   Truck 2 capacity : "+ str(self.Q2)
#        
#        
#        tuples = [(school.pos, "I = "+str(school.inventory),'black') for school in self.S] 
#        tuples = tuples + [(warehouse.pos, "I = "+str(warehouse.inventory),'yellow') for warehouse in self.W]
#        tuples.append((self.central,"CENTRAL", 'black') )
#        
#        
#        layout = dict(
#            title= self.title,
#              annotations= Map.make_annotations(tuples), 
#              font_size=12,
#              showlegend=False,
#              xaxis=axis,
#              yaxis=axis,
#              margin=dict(l=40, r=40, b=85, t=100),
#              hovermode='closest',
#              plot_bgcolor='rgb(248,248,248)',
#              updatemenus= []
#        )
#
#        self.layout = layout
#        self.data = [plot_schools,plot_warehouses, plot_central]
#        self.arrows = [-1,-1,-1]
#        for r in self.R_possible : 
#            number = r.number
#            for x in r.arrows : 
#                self.data.append(x)
#                self.arrows.append(number)
#        
#        for w in self.W :
#            pick_up = w.make_arrow(self.central)
#            self.data.append(pick_up)
#            self.arrows.append(w.name)
#       
#    def make_annotations(tuples):
#        annotations = []
#        for (pos,txt,color) in tuples:
#            annotations.append(
#                dict(
#                    text=txt, # text within the circles
#                    x=pos[0], y=pos[1],
#                    xref='x1', yref='y1',
#                    font=dict(color=color, size=15),
#                    showarrow=False)
#            )
#        return annotations
#
#    def make_updatemenu(self):
#        tuples = [(school.pos, "I = "+str(round(school.inventory,2)),'black') for school in self.S] + [(warehouse.pos, "I = "+str(round(warehouse.inventory,2)),'yellow') for warehouse in self.W]
#        tuples.append( (self.central,"CENTRAL", 'black') )
#        
#        l = len(self.data)
#        visible = [True]*3 + [False]*(l-3)
#
#        if self.t.is_integer():
#            period = " (before lunch)"
#            for k in self.R : 
#                # decide which arrows are visible, 
#                r = self.R_possible[k]
#                for i in range(r.Mk):
#                    e = r.edges[i]
#                    x = (e[0].pos[0]+e[1].pos[0]) / 2 + 1.
#                    y = (e[0].pos[1]+e[1].pos[1]) / 2 + 1.
#                    tuples.append(([x,y], str(round(r.X[i],2))+ "  ", COLORS["road"]))
#
#                i = self.arrows.index(r.number)
#                a = len(r.arrows)
#                visible[i:i+a]=[True]*a
#                
#                
#            for j in self.warehouse_to_deliver : 
#                visible[l-self.N+j ] = True
#        
#        else : 
#            period = "  (evening)"
#            visible[3:] = [False]*(l-3)
#
#
#        annotations = Map.make_annotations(tuples)
#
#        return dict(label="t = "+str(int(self.t))+ period, method = "update", args=[{"visible" : copy(visible)  },{"annotations": annotations, "title":self.title }])
#
#    def run(self,solver,T=10, H=4):
#        '''
#        final function that make the process continue for T days, and plot it into self.fig 
#        Also have to put as an input a function f that will build X and Y
#        '''
#        self.build_Rmat()
#        self.compute_edges()
#        self.compute_costs()
#        
#        t0 = time()
#        X,Y = solver(cost=np.array([r.cost for r in self.R_possible]),
#                q      =np.array([s.q for s in self.S]),
#                C      =np.array([s.C for s in self.S]),
#                I_init = np.array([s.inventory for s in self.S]),
#                r      =self.Rmat,
#                Q=self.Q,T=T,H=H)
#        
#        print("Running time for solver is %f sec" %(time()-t0))
#        self.init_draw()
#        L1,L2 = [],[]
#        for i in range(T):
#            #morning
#            self.select_tours(Y[i])
#            self.compute_X(X[i])
#            self.do_tours()
#            L1.append(self.make_updatemenu())
#            self.R = []
#            self.warehouse_to_deliver = []
#            self.t += 0.5
#            
#            # evening
#            self.eat()
#            L2.append(self.make_updatemenu())
#            self.t += 0.5
#
#
#
#        self.layout["updatemenus"]    = [ dict(buttons = L1, direction = "up",x=0.,y=0.),dict(buttons = L2, direction = "up",x=0.3,y=0.) ]
#        
#
#        self.fig = dict(data=self.data, layout=self.layout)
#
