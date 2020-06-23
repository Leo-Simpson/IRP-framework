#test code

import numpy as np
import pandas as pd

import pulp as plp

n = 10
m = 2
set_I = range(0, n)
set_J = range(0, m)

#random.seed(a=34567)

""" c = {(i,j): random.normalvariate(0,1) for i in set_I for j in set_J}
a = {(i,j): random.normalvariate(0,5) for i in set_I for j in set_J}
l = {(i,j): random.randint(0,10) for i in set_I for j in set_J}
u = {(i,j): random.randint(10,20) for i in set_I for j in set_J}
b = {j: random.randint(0,30) for j in set_J}

 """

np.random.seed(12)
c = np.random.randn(n)
a = 10*np.random.randn(n,m)
b = np.random.randint(10, size = m)
""" 
Problem : min sum_i (2*sum_j(Xij)-1) * Ci s.t. sum_i(Xij Aij) > b: 
 """


opt_model = plp.LpProblem("MIP_Model",plp.LpMinimize)


x_vars = plp.LpVariable.dicts("x",[(i,j) for i in set_I for j in set_J], cat='Binary')


y_vars = { i :
plp.LpVariable(cat=plp.LpInteger, name="y_{0}".format(i)) 
for i in set_I }

y_vars = {i :plp.lpSum(x_vars[i,j] for j in set_J) for i in set_I}

z_vars = {i :y_vars[i]*2 - 1  for i in set_I}


opt_model += plp.lpSum(z_vars[i] * c[i] 
                    for i in set_I ), 'Z'


for j in set_J :
    opt_model += plp.lpSum(a[i,j] * x_vars[i,j] for i in set_I) >= b[j]


for i in set_I : 
    opt_model += z_vars[i] <= 10.


print(c)
print(opt_model)


opt_model.solve(solver = plp.GLPK_CMD())

x = np.zeros((n,m))
z = np.zeros(n)
for i in set_I : 
    z[i] = z_vars[i].value()
    for j in set_J : 
        x[i,j] = x_vars[i,j].varValue
        


print("x",x)
print("z",z)

print(np.sum(a*x,axis=0)-b)
