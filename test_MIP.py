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


opt_model = plp.LpProblem("MIP_Model",plp.LpMinimize)

x_vars  = {(i,j):
plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i,j)) 
for i in set_I for j in set_J}

y_vars = { i :
plp.LpVariable(cat=plp.LpInteger, name="y_{0}".format(i)) 
for i in set_I }

y_vars = {i :plp.lpSum(x_vars[i,j] for j in set_J) for i in set_I}

z_vars = {i :y_vars[i]*2 - 1  for i in set_I}


opt_model += plp.lpSum(z_vars[i] * c[i] 
                    for i in set_I ), 'Z'

for j in set_J :
    opt_model += plp.lpSum(a[i,j] * x_vars[i,j] for i in set_I) <= b[j]

constraints = {j : opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum(a[i,j] * x_vars[i,j] for i in set_I),
             sense=plp.LpConstraintGE,
             rhs=b[j],
             name="constraint{0}".format(j)))
       for j in set_J}



opt_model.solve()

x = np.zeros((n,m))
z = np.zeros(n)
for i in set_I : 
    z[i] = z_vars[i].value()
    for j in set_J : 
        x[i,j] = x_vars[(i,j)].varValue
        

print("x",x)
print("z",z)

print(np.sum(a*x,axis=0)-b)
