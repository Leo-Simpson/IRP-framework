from irp import Warehouse, School, Route, Map
from plotly import offline

import pandas as pd


w1 = Warehouse(position=[0.,0.],capacity = 100, name='w1')
s1 = School(position=[0.,10.], capacity = 5., consumption=1. , name='s1' )
s2 = School(position=[3.,10.], capacity = 3., consumption=3. , name='s2' )
s3 = School(position=[10.,3.], capacity = 2., consumption=2. , name='s3' )
s4 = School(position=[10.,0.], capacity = 4., consumption=1.5, name='s4' )


d = [
    [  0,100,100,100,100],
    [100,  0, 10,150,160],
    [100, 10,  0,140,150],
    [100,150,140,  0, 10],
    [100,160,150, 10,  0]
]



names = ['w1','s1','s2','s3','s4']
D = pd.DataFrame(data=d, columns=names, index=names)

r1 = Route(w1,[s1],   X=[5.])
r2 = Route(w1,[s2],   X=[3.])
r3 = Route(w1,[s3],   X=[2.])
r4 = Route(w1,[s4],   X=[4.])
r5 = Route(w1,[s1,s2],X=[2.,3.])
r6 = Route(w1,[s2,s3],X=[3.,2.])
r7 = Route(w1,[s3,s4],X=[2.,3.])



W = [w1]
S = [s1,s2,s3,s4]
R = [r1,r2,r3,r4,r5,r6,r7]


map = Map(schools=S, warehouses=W, possible_routes=R)
map.run(D,T=10)

offline.plot(map.fig, filename="example1.html")



