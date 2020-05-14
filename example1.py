from drawing import Warehouse, School, Route, Map
from plotly import offline


w1 = Warehouse([0.,0.],100, 'w1')

s1 = School([0.,10.], 5., 1. , 's1')
s2 = School([3.,10.], 3., 3. , 's2' )
s3 = School([10.,3.], 2., 2. , 's3' )
s4 = School([10.,0.], 4., 1.5, 's4')


r1 = Route(w1,[s1,s2],[2.,3.],  costs  = [100,10,100])
r2 = Route(w1,[s3,s4],[2.,3.],  costs  = [100,10,100] )
r3 = Route(w1,[s2,s3],[3.,2.],  costs = [100,140,100] )


W = [w1]
S = [s1,s2,s3,s4]
R = [r1,r2,r3]


map = Map(S, W, possible_routes=R)
map.run()

offline.plot(map.fig, filename="example1.html")



