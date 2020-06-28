import plotly.graph_objects as go 
import numpy as np


from copy import copy, deepcopy


col_vehicules = ['grey', 'lime', 'darksalmon', 'olive', 'violet', 'slategrey' ,'lightgray'] * 10


COLORS = {"school" : "green", "warehouse": "blue", "central":"red","road":col_vehicules}
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
        annotations.append(make_annotation(pos_s[i],"I = {}".format(round(I_s[i],1)),'black'))

    for j in range(len(I_w)):
        annotations.append(make_annotation(pos_w[j],"I = {}".format(round(I_w[j],1)),'yellow'))

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
        if not w["name"] == "CENTRAL" : 
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



def build_arrows(routes):
    # routes is a list TxNxKx[s1,s2,..]
    # arrows is the list of arrows : (i1,i2) with i negative if it represents a warehouse, and indices_step is the list (T,#edges) of indices of arrows  
    indices_step = []
    arrows = []

    def add_indice(start,end,vehicule,l):
        if start != end : 
            try : 
                l.append(arrows.index((start,end,vehicule)))
            except ValueError:
                l.append(len(arrows))
                arrows.append( (start,end,vehicule) )
            

    for t in range(len(routes)):
        l = []
        ind_v = 0
        for n in range(len(routes[0])):
            for k in range(len(routes[0][0])):
                route = routes[t][n][k]
                start = -n-1              # index is negative for the indices of warehouse but positive for schools
                for i_s in route :
                    end = i_s
                    add_indice(start,end,ind_v,l)
                    start = end
                end = -n-1
                add_indice(start,end,ind_v,l)
                if route : ind_v +=1


        indices_step.append(l)

    return arrows, indices_step

def plot_arr(start,end,distance, color):

    # arrows is the list of arrows : ([x1,y1],[x2,y2],distance)
    text = "distance = {}".format(distance)

    x1,y1,x2,y2 = start[0], start[1], end[0], end[1]

    return go.Scatter(x=[(9*x1+x2)/10,(x1+9*x2)/10],
            y=[(9*y1+y2)/10,(y1+9*y2)/10],
            line= dict(color=color),
            text = [text,text],
            hoverinfo='text',
            visible = False
            )





def visu(problem, TITLE, I_s, I_w, km, routes,X):


    title = TITLE + "   Truck 1 capacity : {}   Truck 2 capacity : {} ".format(problem.Q1,problem.Q2)

    pos_s = [s["location"] for s in problem.Schools]
    pos_w = [w["location"] for w in problem.Warehouses]

    data = plots(problem.Schools, problem.Warehouses, problem.central)
    layout = make_layout(title,problem.central,pos_s,pos_w,I_s[0],I_w[0])


    arrows, indices_step = build_arrows(routes) #  arrows is the list of arrows : (i1,i2) with i negative if it represents a warehouse, and indices_step is the list (T,#edges) of indices of arrows  
    Narr = len(arrows)-1
    arrows.extend( [(-n-1, -1) for n in range(1,len(problem.Warehouses)) ])
    for t in range(problem.T):
        for n in range(1,len(problem.Warehouses)):
            if X[t,n]: indices_step[t].append(Narr+n)


    def i_to_dict(i):
        if i>=0 : return(problem.Schools[i])
        else    : return(problem.Warehouses[-i-1])

    for arr in arrows : 
        i1,i2,vehicule = arr 
        start = i_to_dict(i1)
        end   = i_to_dict(i2) 
        color = COLORS["road"][vehicule]
        data.append(plot_arr(start["location"],end["location"],problem.D.loc[start["name"],end["name"]], color))


    L = []
    cumulative_km = 0
    visible_arr = np.zeros(len(arrows), dtype=bool )
    for t in range(problem.T):
        visible_arr[indices_step[t]] = True
        
        cumulative_km += km[t]
        title_up = title + "        KM = {}        Cumulative KM = {} ".format(round(km[t]),round(cumulative_km))
        
        annotations = annotations_inventory( pos_s, pos_w, I_s[t], I_w[t])
        annotations.append(make_annotation(problem.central,"CENTRAL", 'black'))


        step = dict(label="t = "+str(t), method = "update", 
                    args=[{"visible" : [True,True,True]+list(visible_arr)  },
                            {"annotations": annotations, "title":title_up }
                        ]
                    )
        
        L.append(step )
        visible_arr[indices_step[t]] = False
    
    
    layout["updatemenus"]    = [ dict(buttons = L, direction = "up",x=0.,y=0.) ]




    return dict(data=data, layout=layout)







