import plotly.graph_objects as go 
import numpy as np
import time
import numpy.random as rd
from copy import copy 


COLORS = {"school" : "green", "warehouse": "blue"}
TITLE = 'WFP Inventory problem'




class School :
    def __init__(self, position, capacity,consumption,name):
        self.name = name
        self.pos = position
        self.C = capacity
        self.inventory = capacity
        self.q = consumption

    def eat(self):
        self.inventory -= self.q
        
    def receive(self,x):
        self.inventory += x


class Warehouse : 
    def __init__(self, position, capacity,name):
        self.name = name
        self.pos = position
        self.C = capacity
        self.inventory = capacity

    def deliver(self, quantity):
        self.inventory -= quantity

class Route : 
    def __init__(self, warehouse, SCHOOLS):
        self.w = warehouse
        self.S = SCHOOLS
        self.Mk = len(self.S)

    def set_Q(self,Q):
        self.Q = Q      

    def compute_edges(self):
        '''
        Make a list of edges for the route, that is, for each edge, a couple of objects Warehouse or School
        '''
        self.edges = [(self.w,self.S[0])]
        for i in range(self.Mk-1):
            self.edges.append((self.S[i],self.S[i+1]))
        self.edges.append((self.S[-1],self.w))

    def cost(edge,D):
        return D.loc[edge[0].name, edge[1].name]

    def compute_costs(self,D,fixed_costs=0.):
        '''
        compute the cost of a tour with respect to the distances and the fixed cost
        '''
        self.costs = []
        for e in self.edges :
            self.costs.append(Route.cost(e,D))
        
        self.cost = sum(self.costs) + fixed_costs

    def compute_C(self):
        self.room_available = [s.C-s.inventory for s in self.S]
        self.C = min(self.Q, sum(self.room_available))

    def compute_X(self):
        room = self.room_available
        if sum(room) <=self.Q : # we have enough capacity to deliver every one
            self.X = room
        else : 
            # this is the complicate part... 
            food_in_truck = self.Q 
            order = sorted(range(self.Mk), key= lambda i : self.S[i].inventory/self.S[i].q) # we start by filling the schools that nee food the more urgently
            for i in order:
                if food_in_truck > room[i]:
                    self.X[i] = room[i]
                    food_in_truck -=room[i]
                else: 
                    self.X[i] = food_in_truck # note that here, we do tell the truck to go directly home... that could be improved.. 
                    food_in_truck = 0 
                
    def do_tour(self):
        '''
        Change the inventory with respect to a tour and its X
        '''
        self.w.deliver(sum(self.X))
        for i in range(len(self.X)):
            self.S[i].receive(self.X[i])

    def make_arrows(self) : 
        self.arrows= []
        for e in range(self.Mk+1):
            edge = self.edges[e]
            x1, x2, y1, y2 = edge[0].pos[0], edge[1].pos[0], edge[0].pos[1], edge[1].pos[1]
            text = "cost = "+ str(self.costs[e])

            arrow = go.Scatter(x=[(7*x1+x2)/8,(x1+7*x2)/8],
                    y=[(7*y1+y2)/8,(y1+7*y2)/8],
                    line= dict(color="red"),
                    text = [text,text],
                    hoverinfo='text',
                    visible = False
                    )

            self.arrows.append(arrow)

class Map :
    def __init__(self, schools, warehouses, possible_routes = [], Q = 5.):
        '''
        This object represents the state of the problem at some time t 
        '''
        self.S = schools
        self.W = warehouses
        self.R_possible = possible_routes
        for r in self.R_possible : 
            r.set_Q(Q)

        self.M, self.N, self.K = len(schools),len(warehouses), len(possible_routes )
        self.t = 0.
        self.total_cost = 0
        self.cost = 0
        
        for k in range(self.K): 
            self.R_possible[k].number = k

    def build_Rmat(self):
        '''
        Method that build the numpy array Rmat such that r[k,m] = 1 if & only if S[m] in R[k]
        '''
        self.Rmat = np.zeros((self.K,self.M),dtype = bool)
        names = [s.name for s in self.S]
        for k in range(self.K):
            r = self.R_possible[k]
            for stop in r.S : 
                m = names.index(stop.name)
                self.Rmat[k,m] = True

    def compute_edges(self):
        for r in self.R_possible : 
            r.compute_edges()

    def compute_costs(self,D, fixed_costs=0.):
        for r in self.R_possible : 
            r.compute_costs(D, fixed_costs=fixed_costs)

    def select_tours(self):
        '''
        this method will choose which tour we do at this point, and do them (but not plot them, only update the quantities of food)
        the function is not really good fo now, but for the example it should work :) 
        '''
        self.R = []
        L = []
        for m in range(self.M):
            s = self.S[m]
            if (s.inventory < s.q ):  # we only look for the school that need a supply today
                l = np.zeros(self.K, dtype = bool)
                l[self.Rmat[:,m]] = True
                L.append(l)

        
        if L : 
            M = np.array(L)
            r_set = np.array([False]*self.K)
            for i in range(M.shape[1]):
                if sum(M[:,i])>0 : r_set[i] = True
            M = M[:,r_set]
            c = []
            for k in range(self.K):
                if r_set[k]:
                    r = self.R_possible[k]
                    r.compute_C()
                    c.append(r.cost/r.C) # let's consider the cost per quantity of food delivered
            x = np.zeros(self.K, dtype = bool)
            x[r_set] = choose_tours( M, np.array(c) )
            self.R = [ i for i in range(len(x)) if x[i] ]

    def compute_X(self):
        for k in self.R:
            self.R_possible[k].compute_X()

    def do_tours(self):
        for k in self.R :
            r = self.R_possible[k] 
            r.do_tour()
            self.cost += r.cost

        self.total_cost += self.cost
        self.title = TITLE + "        Cost = %s          Total Cost = " %str(self.cost) + str(self.total_cost)
        self.cost = 0

    def eat(self):
        for s in self.S:
            s.eat()

    def init_draw(self):

        # create arrows
        for k in range(self.K): 
            r = self.R_possible[k].make_arrows()

        #plot the schools
        plot_schools = go.Scatter(x=[school.pos[0] for school in self.S],
                        y=[school.pos[1] for school in self.S],
                        mode='markers',
                        name='schools',
                        marker=dict(symbol='circle-dot',
                                        size=50,
                                        color=COLORS["school"]
                                        ),
                        text=[s.name+" C = "+str(s.C)+"  ;   q = "+str(s.q) for s in self.S],
                        hoverinfo='text',
                        opacity=0.8
                        )
        #plot the warehouses
        plot_warehouses = go.Scatter(x=[warehouse.pos[0] for warehouse in self.W],
                            y=[warehouse.pos[1] for warehouse in self.W],
                            mode='markers',
                            name='warehouses',
                            marker=dict(symbol='circle-dot',
                                            size=70,
                                            color=COLORS["warehouse"]
                                            ),
                            text=[w.name+" C = "+str(w.C) for w in self.W],
                            hoverinfo='text',
                            opacity=0.8
                            )

        axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=True,
            showgrid=True,
            showticklabels=True,
            )

        self.title = TITLE + "        Cost = %s          Total Cost = " %str(self.cost) + str(self.total_cost)

        couples = [(school.pos, "I = "+str(school.inventory),'black') for school in self.S] + [(warehouse.pos, "I = "+str(warehouse.inventory),'yellow') for warehouse in self.W]
        layout = dict(
            title= self.title,
              annotations= Map.make_annotations(couples), 
              font_size=12,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)',
              updatemenus= []
        )

        self.layout = layout
        self.data = [plot_schools,plot_warehouses]
        self.arrows = [-1,-1]
        for r in self.R_possible : 
            number = r.number
            for x in r.arrows : 
                self.data.append(x)
                self.arrows.append(number)
       
    def make_annotations(couples):
        annotations = []
        for (pos,txt,color) in couples:
            annotations.append(
                dict(
                    text=txt, # text within the circles
                    x=pos[0], y=pos[1],
                    xref='x1', yref='y1',
                    font=dict(color=color, size=15),
                    showarrow=False)
            )
        return annotations

    def make_updatemenu(self):
        couples = [(school.pos, "I = "+str(school.inventory),'black') for school in self.S] + [(warehouse.pos, "I = "+str(warehouse.inventory),'yellow') for warehouse in self.W]
        l = len(self.data)
        visible = [True]*2 + [False]*(l-2)

        if self.t.is_integer():
            period = " (before lunch)"
            for k in self.R : 
                r = self.R_possible[k]
                for i in range(r.Mk):
                    e = r.edges[i]
                    x = (e[0].pos[0]+e[1].pos[0]) / 2 + .5
                    y = (e[0].pos[1]+e[1].pos[1]) / 2 + .5
                    couples.append(([x,y], str(r.X[i])+ "  ",'red'))

                i = self.arrows.index(r.number)
                a = len(r.arrows)
                visible[i:i+a]=[True]*a
        
        else : 
            period = "  (evening)"
            visible[2:] = [False]*(l-2)


        annotations = Map.make_annotations(couples)

        return dict(label="t = "+str(int(self.t))+ period, method = "update", args=[{"visible" : copy(visible)  },{"annotations": annotations, "title":self.title }])

    def run(self,D ,T=10):
        '''
        final function that make the process continue for T days, and plot it into self.fig 
        '''
        self.build_Rmat()
        self.compute_edges()
        self.compute_costs(D)
        self.init_draw()
        L = []
        for i in range(T):
            #morning
            self.select_tours()
            self.compute_X()
            self.do_tours()
            L.append(self.make_updatemenu())
            self.R = []
            self.t += 0.5
            
            # evening
            self.eat()
            L.append(self.make_updatemenu())
            self.t += 0.5



        self.layout["updatemenus"]    = [ dict(buttons = L, type = "buttons") ]
        

        self.fig = dict(data=self.data, layout=self.layout)






import cvxpy as cp
def choose_tours(M,c):
    '''
    Basically, find x such that Mx >= 1 and the cost is c.x
    So to be clear, we are not sure it is the optimal solution, but at least, we know it is the one that minimize the cost of each day separately
    '''

    v = cp.Variable(len(c), boolean=True)
    constraints = []
    for i in range(len(M)):
        constraints.append(M[i].T@v >= 1)
    problem = cp.Problem(cp.Minimize(c.T @ v), constraints)
    problem.solve()

    return np.array(v.value, dtype = bool)



'''
    def verify_capacity(self,sname):
        for i in range(self.Mk):
            if self.X[i] > (self.S[i].C-self.S[i].inventory) : return False
            if self.S[i].name==sname and self.S[i].q > self.X[i]  : return False
        
        return True
'''

