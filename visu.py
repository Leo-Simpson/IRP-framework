import plotly.graph_objects as go 
import numpy as np
import pandas as pd

from copy import copy, deepcopy



COLORS = {"school" : "green", "warehouse": "blue", "central":"red","road":"grey"}
SIZES = {"school" : 30, "warehouse": 40,"central": 50}


def make_annotation(pos,txt,color):

    return dict(
            text=txt, # text within the circles
            x=pos[0], y=pos[1],
            xref='x1', yref='y1',
            font=dict(color=color, size=15),
            showarrow=False)




def annotations_inventory(pos_s, pos_w, I_s, I_w):
    annotations = []
    for i in range(len(I_s)):
        annotations.append(make_annotation(pos_s[i],"I = {}".format(I_s[i]),'black'))

    for j in range(len(I_w)):
        annotations.append(make_annotation(pos_w[j],"I = {}".format(I_w[j]),'yellow'))

    return annotations


def plots(schools, warehouses, central):
    x, y, text = [],[],[]
    for s in schools:
        
        x.append(s["location"][0])
        y.append(s["location"][1])
        
        t = ""
        for param in ["name","capacity","lower", "consumption", "storage_cost"] :
            t += "{} = {} <br>".format(param,s[param])

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
    
    annotations = annotations_inventory(pos_s,pos_w,I_s,I_w)
    annotations.append(make_annotation(central,"CENTRAL", 'black'))
    



    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
    zeroline=True,
    showgrid=True,
    showticklabels=True,
    )
    
    return dict(
                title= title,
                  annotations= annotations, 
                  font_size=12,
                  showlegend=False,
                  xaxis=axis,
                  yaxis=axis,
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(248,248,248)',
                  updatemenus= []
            )


def visu(problem, TITLE, I_s, I_w, cost, total_cost):
    
    title = TITLE + "   Truck 1 capacity : {}   Truck 2 capacity : {} ".format(problem.Q1,problem.Q2)

    


    pos_s = [s["location"] for s in problem.Schools]
    pos_w = [w["location"] for w in problem.Warehouses]

    data = plots(problem.Schools, problem.Warehouses, problem.central)
    layout = make_layout(title,problem.central,pos_s,pos_w,I_s[0],I_w[0])

    L = []
    visible = [True, True, True]
    for t in range(problem.T):
        visible[-1] = not visible[-1]

        title_up = title + "        Cost = {}        Total Cost = {} ".format(cost[t],total_cost[t])
        
        annotations = annotations_inventory( pos_s, pos_w, I_s[t], I_w[t])
        annotations.append(make_annotation(problem.central,"CENTRAL", 'black'))


        step = dict(label="t = "+str(t), method = "update", 
                    args=[{"visible" : copy(visible)  },
                            {"annotations": annotations, "title":title_up }
                        ]
                    )

        L.append(step )
    
    
    layout["updatemenus"]    = [ dict(buttons = L, direction = "up",x=0.,y=0.) ]




    return dict(data=data, layout=layout)



'''

def make_updatemenu(self):
        tuples = [(school.pos, "I = "+str(round(school.inventory,2)),'black') for school in self.S] + [(warehouse.pos, "I = "+str(round(warehouse.inventory,2)),'yellow') for warehouse in self.W]
        tuples.append( (self.central,"CENTRAL", 'black') )
        
        l = len(self.data)
        visible = [True]*3 + [False]*(l-3)

        if self.t.is_integer():
            period = " (before lunch)"
            for k in self.R : 
                # decide which arrows are visible, 
                r = self.R_possible[k]
                for i in range(r.Mk):
                    e = r.edges[i]
                    x = (e[0].pos[0]+e[1].pos[0]) / 2 + 1.
                    y = (e[0].pos[1]+e[1].pos[1]) / 2 + 1.
                    tuples.append(([x,y], str(round(r.X[i],2))+ "  ", COLORS["road"]))

                i = self.arrows.index(r.number)
                a = len(r.arrows)
                visible[i:i+a]=[True]*a
                
                
            for j in self.warehouse_to_deliver : 
                visible[l-self.N+j ] = True
        
        else : 
            period = "  (evening)"
            visible[3:] = [False]*(l-3)


        annotations = Map.make_annotations(tuples)

        return dict(label="t = "+str(int(self.t))+ period, method = "update", args=[{"visible" : copy(visible)  },{"annotations": annotations, "title":self.title }])


'''