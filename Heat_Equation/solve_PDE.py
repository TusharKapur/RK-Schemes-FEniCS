"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0

  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import calculate_f

set_log_level(30)

def solve_PDE(method, T, num_steps):

  #T = 1.0            # final time
  #num_steps = 200     # number of time steps
  dt = T / num_steps # time step size
  alpha = 3          # parameter alpha
  beta = 1.2         # parameter beta
  gamma = 0          # parameter gamma (controls time dependence)

  iteration = 0
  approx_error = 0.0
  upper_tol = 1
  set_Diverged = 0

  #method = "RK4"
  #method = "HEUN"
  #method = "EXPLICIT_EULER"

  # Create mesh and define function space
  nx = ny = 4
  mesh = UnitSquareMesh(nx, ny)
  V = FunctionSpace(mesh, 'P', 1)

  #Define initial condition # Not Used  #TODO: Remove
  #u_init = Expression ( '1 - (x[0]-1)*(x[0]-1) - (x[1]-1)*(x[1]-1)', degree = 2 )
  #u_init = Expression ( '0.0', degree = 2 )

  # Define boundary condition
  u_D = Expression('1 + alpha*x[0]*x[0] + beta*x[1]*x[1] + pow((1+t), gamma)',
                    degree=2, alpha=alpha, beta=beta, gamma=gamma, t=0)
  #u_D = Expression('0',
  #                  degree=2, alpha=alpha, beta=beta, gamma=gamma, t=0)

  (f, f_np05, _, f_np1) = calculate_f.get_problem_setup(alpha, beta, gamma)

  def boundary(x, on_boundary):
      return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = interpolate(u_D, V)
  u1 = interpolate(u_D, V)
  u2 = interpolate(u_D, V)
  u3 = interpolate(u_D, V)
  #u_n = project(u_D, V)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  #f = Constant(beta - 2 - 2*alpha) # Already defined

  a = inner(u, v)*dx

  # Time-stepping
  u = Function(V)
  t = 0
  for n in range(num_steps):

      # Update current time
      # t += dt  # moved this down. Otherwise your boundary conditions are ahead of time. Remeber: You are working with explicit schemes here.
      u_D.t = t
      f_n.t = t  # - dt  , not needed, since we not use t_n
      f_np05.t = t + 0.5*dt  # - 0.5*dt  , not needed, since we not use t_n
      # f_np05.t = t + 0.5*dt  # - 0.5*dt  , not needed, since we not use t_n; identical to the one above.
      f_np1.t = t + dt

      # Calculate RHS and solve
      if method == "EXPLICIT_EULER":  #Explicit Euler
        L = inner(u_n, v)*dx - dt*inner(grad(u_n), grad(v))*dx + dt*inner(f, v)*dx        
        # for u (= u_n+1) you need the bc at t_n+1, for L you need a bc at u_n, since it is explicit.
        # Did you already try out using bc.apply()? I'm not sure whether solve with a bc is the right approach for explicit time-stepping.
        # bc.apply(L, 0)
        # u_D.t = t + dt  # to get bc at t_n+1 that has to be applied to u?
        solve(a == L, u, bc)        
        # rest for Explicit Euler looks good to me. the solve is basically just invetring the mass matrix in "a" and computes u (= u_np1)

      elif method == "HEUN":  #Heun's Method

        #TODO: Are the BCS updated correctly? (Also applies to RK4)
        L1 = inner(u_n, v)*dx - dt*inner(grad(u_n), grad(v))*dx + dt*inner(f_n, v)*dx #Calculation of y'~_{n+1}
        # I think that we again need a bc.apply() here to catch the explicit part correctly.
        solve(a == L1, u1, bc) #Calculation of y~_{n+1}
        L2 = inner(u1, v)*dx - dt*inner(grad(u1), grad(v))*dx + dt*inner(f_np1, v)*dx
        L = 0.5*(L1 + L2)        
        solve(a == L, u, bc)  #Calculation of y_{n+1}

      elif method == "RK4":  #4th Order Runge-Kutta's Method

        L1 = inner(u_n, v)*dx - dt*inner(grad(u_n), grad(v))*dx + dt*inner(f, v)*dx
        solve(a == L1, u1, bc)
        L2 = inner(u1, v)*dx - 0.5*dt*inner(grad(u1), grad(v))*dx + 0.5*dt*inner(f_np1, v)*dx
        solve(a == L2, u2, bc)
        L3 = inner(u2, v)*dx - 0.5*dt*inner(grad(u2), grad(v))*dx + 0.5*dt*inner(f_np2, v)*dx
        solve(a == L3, u3, bc)
        L4 = inner(u3, v)*dx - dt*inner(grad(u3), grad(v))*dx + dt*inner(f_np3, v)*dx
        L = (1/6)*(L1 + 2*L2 + 2*L3 + L4)
        solve(a == L, u, bc)    

      # Plot solution
      #plot(u)

      # Compute error at vertices
      u_e = interpolate(u_D, V)
      error = np.abs(np.array(u_e.vector()) - np.array(u.vector())).max()
      approx_error += error*error # Calculation continued later
      #print('t = %.2f: error = %.3g' % (t, error))
      if error > upper_tol:
        print("%s scheme diverged for timestep size: %.3g" %(method, dt))
        set_Diverged = 1
        approx_error = "NaN"
        break

      # Update previous solution
      u_n.assign(u)

      iteration += 1
      t += dt

  if type(approx_error) is not str:
    approx_error =  sqrt(approx_error*dt/T)
    print(iteration, approx_error, method)

  # Hold plot
  #fig = plt.figure(figsize=(10,5))
  #plot(u)
  #fig.savefig('plot.jpg', bbox_inches='tight', dpi=150)
  #plt.show()

  return approx_error
