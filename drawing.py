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
    def __init__(self, warehouse, SCHOOLS, X, costs = None):
        self.w = warehouse
        self.S = SCHOOLS
        self.X =X
        self.Mk = len(self.S)

        if costs is None : self.costs = [0]*(self.Mk+1)
        else : self.costs = costs
        self.cost = sum(costs)

        food = sum(X)
        self.C = food
        self.food_in_truck = [food]
        self.edges = [(warehouse,SCHOOLS[0])]
        for i in range(self.Mk-1):
            self.edges.append((self.S[i],self.S[i+1]))
            self.food_in_truck.append(food-X[i])
        self.edges.append((self.S[-1],warehouse))
        self.food_in_truck.append(0)

    def do_tour(self):
        self.w.deliver(self.C)
        for i in range(len(self.X)):
            self.S[i].receive(self.X[i])

    def verify_capacity(self):
        for i in range(self.Mk):
            if self.X[i] > (self.S[i].C-self.S[i].inventory) : return False
        return True

    def make_data(self) : 
        self.data= []
        for e in range(self.Mk+1):
            edge = self.edges[e]
            x1, x2, y1, y2 = edge[0].pos[0], edge[1].pos[0], edge[0].pos[1], edge[1].pos[1]
            text1 = str(self.food_in_truck[e]) +" in the truck"
            text2 = "cost = "+ str(self.costs[e])

            arrow = go.Scatter(x=[(7*x1+x2)/8,(x1+7*x2)/8],
                    y=[(7*y1+y2)/8,(y1+7*y2)/8],
                    line= dict(color="red"),
                    text = [text1+"  ;  "+text2,text1+"  ;  "+text2],
                    hoverinfo='text',
                    visible = False
                    )

            self.data.append(arrow)



class Map :
    def __init__(self, schools, warehouses, possible_routes = [], t=0.):
        '''
        This object represents the state of the problem at some time t 
        '''
        self.S = schools
        self.M = len(schools)
        self.W = warehouses
        self.N = len(warehouses)
        self.R_possible = sorted(possible_routes, key=lambda x: float(x.cost)/x.C)
        self.K = len(possible_routes )
        self.t = t
        self.total_cost = 0
        self.cost = 0
        self.title = TITLE + "        Cost = %s          Total Cost = " %str(self.cost) + str(self.total_cost)
        self.build_Rmat()
        for k in range(self.K): 
            r = self.R_possible[k]
            r.make_data()
            r.number = k


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

    def choose_tours(self):
        '''
        this method will choose which tour we do at this point, and do them (but not plot them, only update the quantities of food)
        the function is not really good fo now, but for the example it should work :) 
        '''
        self.R = []
        for m in range(self.M):
            s = self.S[m]
            name = s.name
            if (s.inventory >= s.q ): continue # we only look for the school that need a supply today
            route_indices = np.arange(self.K)[self.Rmat[:,m]]
            L = [self.R_possible[k] for k in route_indices ] # list of the possible routes that serve s

            # I don't know how to choose the right route for now... 
            # let's take the first one possible for the example, which means that routes are already ordered by preference
            for k in range(len(L)) :
                r = L[k]
                if r.verify_capacity():break

            self.R.append(r)
            self.cost += r.cost
            r.do_tour()
        self.total_cost += self.cost
        self.title = TITLE + "        Cost = %s          Total Cost = " %str(self.cost) + str(self.total_cost)
        self.cost = 0

    def eat(self):
        for s in self.S:
            s.eat()

    def init_draw(self):

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


        layout = dict(
            title= self.title,
              annotations= self.make_annotations(), 
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
            for x in r.data : 
                self.data.append(x)
                self.arrows.append(number)

        

    def make_annotations(self):
        couples = [(school.pos, "I = "+str(school.inventory)) for school in self.S] + [(warehouse.pos, "I = "+str(warehouse.inventory)) for warehouse in self.W]
        annotations = []
        for (pos,txt) in couples:
            annotations.append(
                dict(
                    text=txt, # text within the circles
                    x=pos[0], y=pos[1],
                    xref='x1', yref='y1',
                    font=dict(color='black', size=15),
                    showarrow=False)
            )
        return annotations

    def make_updatemenu(self):

        annotations = self.make_annotations()
        l = len(self.data)
        visible = [True]*2 + [False]*(l-2)


        if self.t.is_integer():
            period = " (before lunch)"
            for r in self.R : 
                i = self.arrows.index(r.number)
                a = len(r.data)
                visible[i:i+a]=[True]*a
        
        else : 
            period = "  (evening)"
            visible[2:] = [False]*(l-2)

        


        args = [
            {"visible" : copy(visible)  },
            {"annotations": annotations, "title":self.title }
            ]

        return dict(label="t = "+str(int(self.t))+ period, method = "update", args=args)

    def run(self, T=10):
        '''
        final function that make the process continue for T days, and plot it into self.fig 
        '''
        self.init_draw()
        L = []
        for i in range(T):
            #morning
            self.choose_tours()
            L.append(self.make_updatemenu())
            self.R = []
            self.t += 0.5
            
            # evening
            self.eat()
            L.append(self.make_updatemenu())
            self.t += 0.5



        self.layout["updatemenus"]    = [ dict(buttons = L, type = "buttons") ]
        

        self.fig = dict(data=self.data, layout=self.layout)





                            
                            


