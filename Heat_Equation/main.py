#TODO: Make tables
from dolfin.fem.solving import _solve_varproblem
import solve_PDE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import os

methods = ("RK4", "HEUN", "EXPLICIT_EULER")
T = 1.0
#steps = [80, 100, 120, 140, 160]
steps = np.linspace(80, 640, 29)
dt = T/steps
dt = np.flip(dt)
steps = steps.astype(int)

i = 0
shp = (np.size(steps),np.size(methods))
errors = np.empty(shp)
errors[:,:] = "NaN"

for i in range(len(steps)):
    for j in range(len(methods)):
        approx_error = solve_PDE.solve_PDE(methods[j], T, steps[i])
        errors[i, j] = approx_error

# val = [["" for c in range(errors.shape[1])] for r in range(errors.shape[0])] 
# fig1, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = errors,  
#     rowLabels = steps,  
#     colLabels = methods, 
#     rowColours =["palegreen"] * 10,  
#     colColours =["palegreen"] * 10, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# ax.set_title('Approximation Errors', 
#              fontweight ="bold")
# plt.show() 
# fig1.savefig('table.jpg', bbox_inches='tight', dpi=150)
# os.system('wslview table.jpg')

style.use('ggplot') 
fig2 = plt.figure()
axes = fig2.add_axes([0.1,0.1,0.8,0.8])
axes.set_ylim([0,1e-13])    # When gamma = 0
#axes.set_ylim([0,1e-01])   # When gamma = 1

xx = dt
plt.plot(xx, errors[:,0], 'b', linewidth=2)
plt.plot(xx, errors[:,1], 'r', linewidth=2)
plt.plot(xx, errors[:,2], 'g', linewidth=2)
plt.legend(["Runge Kutta 4", "Heun", "Explicit Euler"])
plt.xlabel("dt")
plt.ylabel("Approximation Error")

fig2.savefig('plot.jpg', bbox_inches='tight', dpi=150)
os.system('wslview plot.jpg')