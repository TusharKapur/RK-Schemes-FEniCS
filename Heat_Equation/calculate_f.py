#from __future__ import print_function
from fenics import *
import numpy as np
import sympy as sp

def get_manufactured_solution(alpha, beta, gamma):
    x, y, t, dt = sp.symbols('x[0], x[1], t, dt')
    manufactured_solution = 1 + alpha * x**2 + beta * y**2 + pow((1+t), gamma)
    #manufactured_solution = 0
    #print("manufactured solution = {}".format(manufactured_solution))
    return manufactured_solution
    
def get_problem_setup(alpha, beta, gamma):
    x, y, t, dt = sp.symbols('x[0], x[1], t, dt')
    u_analytical = get_manufactured_solution(alpha, beta, gamma)
    u_D = Expression(sp.printing.ccode(u_analytical), degree=2, t=0)

    f_rhs = - sp.diff(sp.diff(u_analytical, x), x) - sp.diff(sp.diff(u_analytical, y), y) + sp.diff(u_analytical, t)
    f = Expression(sp.printing.ccode(f_rhs), degree=2, t=0)
    f_np1 = Expression(sp.printing.ccode(f_rhs), degree=2, t=0)
    f_np2 = Expression(sp.printing.ccode(f_rhs), degree=2, t=0)
    f_np3 = Expression(sp.printing.ccode(f_rhs), degree=2, t=0)
    return (f, f_np1, f_np2, f_np3)