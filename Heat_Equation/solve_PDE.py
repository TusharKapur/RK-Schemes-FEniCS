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
  gamma = 1.0          # parameter gamma (controls time dependence)

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

  (f, f_np1, f_np2, f_np3) = calculate_f.get_problem_setup(alpha, beta, gamma)

  def boundary(x, on_boundary):
      return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  def LeastSquare(x, y):
    from numpy import linalg
    result = linalg.lstsq(x, y)[0]
    return result

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
      f_np1.t = t
      f_np2.t = t + 0.5*dt
      f_np3.t = t + 0.5*dt
      f.t = t + dt

      # Calculate RHS and solve
      if method == "EXPLICIT_EULER":  #Explicit Euler

        u_D.t = t + dt
        bc = DirichletBC(V, u_D, boundary)
        L = inner(u_n, v)*dx - dt*inner(grad(u_n), grad(v))*dx + dt*inner(f, v)*dx
        A = assemble(a)
        b = assemble(L)
        bc.apply(A, b)
        u = Function(V)
        u.vector()[:] = LeastSquare(A.array(), b.get_local())
        # solve(a == L, u, bc)

      elif method == "HEUN":  #Heun's Method

        # Note: In Heun's method, we are solving for u at "t+dt" in both the steps. So, BC is updated to "t+dt" for these steps.
        u_D.t = t + dt
        bc = DirichletBC(V, u_D, boundary)
        L1 = inner(u_n, v)*dx - dt*inner(grad(u_n), grad(v))*dx + dt*inner(f_np1, v)*dx
        slope1 = - inner(grad(u_n), grad(v))*dx + inner(f_np1, v)*dx #Calculation of y'~_{n+1}
        A = assemble(a)
        b = assemble(L1)
        bc.apply(A, b)
        u1 = Function(V)
        u1.vector()[:] = LeastSquare(A.array(), b.get_local())        
        # solve(a == L1, u1, bc) #Calculation of y~_{n+1}
        
        u_D.t = t + dt
        bc = DirichletBC(V, u_D, boundary)
        L2 = inner(u_n, v)*dx - dt*inner(grad(u1), grad(v))*dx + dt*inner(f, v)*dx
        slope2 = - inner(grad(u1), grad(v))*dx + inner(f, v)*dx
        # L1 -= inner(u_n, v)*dx
        # L2 -= inner(u1, v)*dx
        #L = 0.5*(L1 + L2)
        slope = 0.5*(slope1 + slope2)
        L = inner(u_n, v)*dx + dt*slope
        A = assemble(a) # TODO: Repeatedly assembling A may not be necessary (also in RK4)
        b = assemble(L)
        bc.apply(A, b)
        u = Function(V)
        u.vector()[:] = LeastSquare(A.array(), b.get_local())
        # solve(a == L, u, bc)  #Calculation of y_{n+1}

      elif method == "RK4":  #4th Order Runge-Kutta's Method
        
        # Note: In RK4 method, we are solving for u at "t+dt" in the 1st and 4th steps. So, BC is updated to "t+dt" for these steps.
        # In 2nd and 3rd steps, we are solving for u at "t + dt/2". So, BC is updated to "t + dt/2" for these steps.
        u_D.t = t + dt  
        bc = DirichletBC(V, u_D, boundary)
        L1 = inner(u_n, v)*dx - dt*inner(grad(u_n), grad(v))*dx + dt*inner(f_np1, v)*dx
        slope1 = - inner(grad(u_n), grad(v))*dx + inner(f_np1, v)*dx
        A = assemble(a)
        b = assemble(L1)
        bc.apply(A, b)
        u1 = Function(V)
        u1.vector()[:] = LeastSquare(A.array(), b.get_local())
        # solve(a == L1, u1, bc)
        
        u_D.t = t + 0.5*dt
        bc = DirichletBC(V, u_D, boundary)
        L2 = inner(u_n, v)*dx - 0.5*dt*inner(grad(u1), grad(v))*dx + 0.5*dt*inner(f_np2, v)*dx
        slope2 = - inner(grad(u1), grad(v))*dx + inner(f_np2, v)*dx
        A = assemble(a)
        b = assemble(L2)
        bc.apply(A, b)
        u2 = Function(V)
        u2.vector()[:] = LeastSquare(A.array(), b.get_local())
        # solve(a == L2, u2, bc)
        
        u_D.t = t + 0.5*dt
        bc = DirichletBC(V, u_D, boundary)
        L3 = inner(u_n, v)*dx - 0.5*dt*inner(grad(u2), grad(v))*dx + 0.5*dt*inner(f_np3, v)*dx
        slope3 = - inner(grad(u2), grad(v))*dx + inner(f_np3, v)*dx
        A = assemble(a)
        b = assemble(L3)
        bc.apply(A, b)
        u3 = Function(V)
        u3.vector()[:] = LeastSquare(A.array(), b.get_local())
        # solve(a == L3, u3, bc)
        
        u_D.t = t + dt
        bc = DirichletBC(V, u_D, boundary)
        L4 = inner(u_n, v)*dx - dt*inner(grad(u3), grad(v))*dx + dt*inner(f, v)*dx
        slope4 = - inner(grad(u3), grad(v))*dx + inner(f, v)*dx
        #L = (1/6)*(L1 + 2*L2 + 2*L3 + L4)     
        slope = (1/6)*(slope1 + 2*slope2 + 2*slope3 + slope4)
        L = inner(u_n, v)*dx + dt*slope
        A = assemble(a)
        b = assemble(L)
        bc.apply(A, b)
        u = Function(V)
        u.vector()[:] = LeastSquare(A.array(), b.get_local())
        # solve(a == L, u, bc)    

      # Plot solution
      #plot(u)

      # Compute error at vertices
      u_e = interpolate(u_D, V)
      error = np.abs(np.array(u_e.vector()) - np.array(u.vector())).max()
      # approx_error += error*error # Calculation continued later
      #print('t = %.2f: error = %.3g' % (t, error))
      if error > upper_tol:
        print("%s scheme diverged for timestep size: %.3g" %(method, dt))
        set_Diverged = 1
        error = "NaN"
        break

      # Update previous solution
      u_n.assign(u)

      iteration += 1
      t += dt

  if type(error) is not str:
    # approx_error =  sqrt(approx_error*dt/T)
    print(iteration, error, method)

  # Hold plot
  #fig = plt.figure(figsize=(10,5))
  #plot(u)
  #fig.savefig('plot.jpg', bbox_inches='tight', dpi=150)
  #plt.show()

  return error
