# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:56:59 2019

@author: Jasper
"""
from sympy.solvers import solve
from sympy import Symbol
x = Symbol('x')
t = -5
print(solve(x**2 - 1 + t, x))