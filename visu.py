import plotly.graph_objects as go 
import numpy as np


from copy import copy, deepcopy


'''
This files' main purpose is to build the function visu, that create a visualisation of a solution. 
This visualisation is made thanks to the 

'''




col_vehicules = ['grey', 'lime', 'darksalmon', 'olive', 'violet', 'slategrey' ,'lightgray'] * 10
col_vehicules.append("black") # color for the pickups

COLORS = {"school" : "green", "warehouse": "blue", "central":"red","road":col_vehicules}
SIZES = {"school" : 30, "warehouse": 40,"central": 50}

annotation_size = 10


def make_annotation(pos,txt,color):

    return dict(
            text=txt, # text within the circles
            x=pos[0], y=pos[1],
            xref='x1', yref='y1',
            font=dict(color=color, size=annotation_size),
            showarrow=False)




def annotations_inventory(pos_s, pos_w, I_s, I_w):
    annotations = []
    for i in range(len(I_s)):
        annotations.append(make_annotation(pos_s[i],str(round(I_s[i],1)),'black'))

    for j in range(len(I_w)):
        annotations.append(make_annotation(pos_w[j],str(round(I_w[j],1)),'yellow'))

    return annotations


def plots(schools, warehouses,central,central_name):
    x, y, text = [],[],[]
    for s in schools:
        
        x.append(s["location"][0])
        y.append(s["location"][1])
        
        t = "{} : {} <br>".format("name",s['name'])
        for param in ["capacity","lower", "consumption", "storage_cost"] :
            t += "{} = {} <br>".format(param,round(s[param],4))

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
    for i,w in enumerate(warehouses):
        if i>0 :
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
                                text=[central_name],
                                hoverinfo='text',
                                opacity=0.8
                                )
    return [plot_s, plot_w, plot_c]



def make_layout(title,central, pos_s, pos_w, I_s, I_w): 
    
    annotations = annotations_inventory(pos_s,pos_w,I_s,I_w)
    #annotations.append(make_annotation(central,"CENTRAL", 'black'))
    



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



def build_arrows(routes, q, makes):
    # routes is a list TxNxKx[s1,s2,..]
    # arrows is the list of arrows : (i1,i2, v) with i negative if it represents a warehouse, and indices_step is the list (T,#edges) of indices of arrows  
    indices_step = []
    arrows = []

    def add_indice(start,end,vehicule,l,quant,name):
        if start != end : 
            l.append(len(arrows))
            arrows.append( (start,end,vehicule, quant, name) )
    
    
            

    for t in range(len(routes)):
        l = []
        ind_v = 0
        for n in range(len(routes[0])):
            for k in range(len(routes[0][0])):
                route = routes[t][n][k]
                name = makes[n,k]
                if route : 
                    quantity = q[t,n,k]
                    start = -n-1              # index is negative for the indices of warehouse but positive for schools
                    add_indice(route[-1],-n-1,ind_v,l,0,name) # last arrow
                    for i_s in route :
                        end = i_s
                        add_indice(start,end,ind_v,l,quantity[i_s],name)
                        start = end
                    
                    ind_v +=1

        indices_step.append(l)


    return arrows, indices_step

def plot_arr(start,end,distance, color, quantity, name):


    # arrows is the list of arrows : ([x1,y1],[x2,y2],distance)
    text = name + " <br> distance = {}".format(round(distance,2)) + " <br> " + "quantity for next school = {}".format(round(quantity,3)) 

    x1,y1,x2,y2 = start[0], start[1], end[0], end[1]

    return go.Scatter(x=[(9*x1+x2)/10,(x1+9*x2)/10],
            y=[(9*y1+y2)/10,(y1+9*y2)/10],
            line= dict(color=color),
            text = [text,text],
            hoverinfo='text',
            visible = False
            )





def visu(schools, warehouses, TITLE, I_s, I_w, km, routes1,X, q, Q2, D,makes, time_step):
    '''
        Main function, that build a dict, compatible with the plotly.figure object , that contain all the data of the visualisation of the solution

        Args : 
            schools (list): list of dictionary {'capacity': ..., 'lower':..., 'consumption': ...,'storage_cost': ... , 'initial': ...,  'name' : ..., 'location':...}
            Warehouse (list): list of dictionary {"capacity": .., "lower":.., "dist_central":.., "fixed_cost":.., "initial": .., "name": ..., "location": ... }
                It can contains as well the central warehouse in the first position. 
            TITLE (string): title to be print at the top of the animation. 
            I_s (list) : list of school inventories, for each time step
            I_w (list) : list of warehouse inventories, for each time step
            km (list of float) : list of the total km driven for each time step
            routes1 (list): each r[t][n][k] is the ordered list of the schools visited in a tour
            X (np.ndarray): T+1xN array with boolean true when warehouse is supplied at time t
            q (np.ndarray): T+1xNxKxM array with portion of food delivered to school m by vehicle k of warehouse n at time t
            Q2 (int): Capacity of the second type of vehicle : the one that serve the warehouses.
            D (np.ndarray): Matrix of distances. If None, the geodesic distance will be taken.
            makes (np.array, optional) : NxK matrix with the string of the names of the vehicles. 
            time_step (float) : length of a time step in weeks 


    '''

    N,T =len(warehouses),len(I_s)
    routes = [[[ [routes1[t][n][k][m]-N for m in range(len(routes1[t][n][k]))] for k in range(len(routes1[t][n]))] for n in range(N)] for t in range(T)]
    

    #title = TITLE + "   Truck 1 capacity : {}   Truck 2 capacity : {} ".format(problem.Q1,problem.Q2)
    title = TITLE 

    


    pos_s = [s["location"] for s in schools]
    pos_w = [w["location"] for w in warehouses]
    central = warehouses[0]["location"]
    central_name = warehouses[0]["name"]

    data = plots(schools, warehouses, central, central_name)
    layout = make_layout(title,central,pos_s,pos_w,I_s[0],I_w[0])


    arrows, indices_step = build_arrows(routes, q,makes) #  arrows is the list of arrows : (i1,i2) with i negative if it represents a warehouse, and indices_step is the list (T,#edges) of indices of arrows  
    Narr = len(arrows)-1
    arrows.extend( [(-n-1, -1,-1,Q2,'pickup') for n in range(1,N) ])
    for t in range(T):
        for n in range(1,N):
            if X[t,n]: indices_step[t].append(Narr+n)


    def i_to_dict(i):
        if i>=0 : return(schools[i], i+N)
        else    : return(warehouses[-i-1], -i-1)

    for arr in arrows : 
        i1,i2,vehicule, quantity, name = arr 
        start,ind1 = i_to_dict(i1)
        end,ind2   = i_to_dict(i2) 
        color = COLORS["road"][vehicule]
        data.append(plot_arr(start["location"],end["location"],D[ind1,ind2], color, quantity, name))


    L = []
    cumulative_km = 0
    visible_arr = np.zeros(len(arrows), dtype=bool )
    for t in range(T):
        visible_arr[indices_step[t]] = True
        
        cumulative_km += km[t]
        title_up = title + "        KM = {}        Cumulative KM = {} ".format(round(km[t]),round(cumulative_km))
        
        annotations = annotations_inventory( pos_s, pos_w, I_s[t], I_w[t])
        #annotations.append(make_annotation(problem.central,"CENTRAL", 'black'))


        step = dict(label="t = "+str(t), method = "update", 
                    args=[{"visible" : [True,True,True]+list(visible_arr)  },
                            {"annotations": annotations, "title":title_up }
                        ]
                    )
        
        L.append(step )
        visible_arr[indices_step[t]] = False
    
    
    layout["updatemenus"]    = [ dict(buttons = L, direction = "up",x=0.,y=0.) ]




    return dict(data=data, layout=layout)







