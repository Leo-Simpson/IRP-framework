from irp import Warehouse, School, Route, Map
from plotly import offline

import pandas as pd


w1 = Warehouse(position=[0.,0.],capacity = 100, name='w1')
s1 = School(position=[0.,10.], capacity = 5., consumption=1. , name='s1' )
s2 = School(position=[3.,10.], capacity = 3., consumption=3. , name='s2' )
s3 = School(position=[10.,3.], capacity = 2., consumption=2. , name='s3' )
s4 = School(position=[10.,0.], capacity = 4., consumption=1.5, name='s4' )

r1 = Route(w1,[s1])
r2 = Route(w1,[s2])
r3 = Route(w1,[s3])
r4 = Route(w1,[s4])
r5 = Route(w1,[s1,s2])
r6 = Route(w1,[s2,s3])
r7 = Route(w1,[s3,s4])



# distance matrix
d = [
    [  0,100,100,100,100],
    [100,  0, 10,150,160],
    [100, 10,  0,140,150],
    [100,150,140,  0, 10],
    [100,160,150, 10,  0]
]

# transform it to a dataframe
names = ['w1','s1','s2','s3','s4']
D = pd.DataFrame(data=d, columns=names, index=names)





map = Map(
    schools=[s1,s2,s3,s4],
    warehouses=[w1],
    possible_routes=[r1,r2,r3,r4,r5,r6,r7],
    Q = 5.
    )


map.run(D,T=10)


offline.plot(map.fig, filename="example1.html")



