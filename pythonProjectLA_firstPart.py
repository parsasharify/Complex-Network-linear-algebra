import copy
import math
import json
import numpy as np
import networkx as nx
import scipy as sp
from scipy.integrate import RK45
from matplotlib import pyplot as plt
from math import e
from scipy import integrate as inte
def f(t,y):
    return math.sqrt(abs(y))



solution = inte.RK45(f, 2 , [1] , 10 , 0.1, e**-6)

# collect data
t_values = []
y_values = []
for i in range(80):
    # get solution step state
    solution.step()
    t_values.append(solution.t)
    y_values.append(solution.y[0])
    # break loop after modeling is finished
    if solution.status == 'finished':
        break

solution = inte.RK45(f, 2 , [1] , 0 , 0.1, e**-6)

for i in range(80):
    # get solution step state
    solution.step()
    t_values.append(solution.t)
    y_values.append(solution.y[0])
    # break loop after modeling is finished
    if solution.status == 'finished':
        break

t_values, y_values = zip(*(sorted(zip(t_values,y_values))))
print(t_values)
print(y_values)
plt.plot(t_values, y_values)
plt.show()


