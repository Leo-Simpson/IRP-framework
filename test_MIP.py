#test code

import numpy as np
import pandas as pd

import pulp as plp

n = 100
m = 20
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
c = np.random.randn(n,m)
a = 10*np.random.randn(n,m)
l = np.random.randint(10, size = (n,m))
u = np.random.randint(10, high=20, size = (n,m))
b = np.random.randint(10, size = m)






opt_model = plp.LpProblem(name="MIP_Model")

# if x is Continuous
""" x_vars  = {(i,j):
plp.LpVariable(cat=plp.LpContinuous, 
               lowBound=l[i,j], upBound=u[i,j], 
               name="x_{0}_{1}".format(i,j)) 
for i in set_I for j in set_J} """
# if x is Binary
x_vars  = {(i,j):
plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i,j)) 
for i in set_I for j in set_J}
# if x is Integer
""" x_vars  = {(i,j):
plp.LpVariable(cat=plp.LpInteger, 
               lowBound=l[i,j], upBound= u[i,j],
               name="x{0}{1}".format(i,j)) 
for i in set_I for j in set_J}
 """

# Less than equal constraints
""" constraints = {j : opt_model.addConstraint(
                            plp.LpConstraint(
                                        e=m(a[i,j] * x_vars[i,j] for i in set_I),
                                        sense=plp.plp.LpConstraintLE,
                                        rhs=b[j],
                                        name="constraint_{0}".format(j)))
                for j in set_J} """
# >= constraints
constraints = {j : opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum(a[i,j] * x_vars[i,j] for i in set_I),
             sense=plp.LpConstraintGE,
             rhs=b[j],
             name="constraint{0}".format(j)))
       for j in set_J}

""" # == constraints
constraints = {j : opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum(a[i,j] * x_vars[i,j] for i in set_I),
             sense=plp.LpConstraintEQ,
             rhs=b[j],
             name="constraint_{0}".format(j)))
       for j in set_J} """



objective = plp.lpSum(x_vars[i,j] * c[i,j] 
                    for i in set_I 
                    for j in set_J)



# for minimization
opt_model.sense = plp.LpMinimize
opt_model.setObjective(objective)


# solving with CBC
#opt_model.solve()
# solving with Glpk
opt_model.solve(solver = plp.GLPK_CMD())

x = np.zeros((n,m))
for i in set_I : 
    for j in set_J : 
        x[i,j] = x_vars[(i,j)].varValue

print(x)

print(np.sum(a*x,axis=0)-b)
