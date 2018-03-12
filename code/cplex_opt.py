
#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: lpex2.py
# Version 12.6.3
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2015. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
#
# lpex2.py - Reading and optimizing a problem.  Demonstrates
           # specifying optimization method by setting parameters.
#
# The user has to choose the method on the command line:
#
#    python lpex2.py <filename> o     cplex default
#    python lpex2.py <filename> p     primal simplex
#    python lpex2.py <filename> d     dual simplex
#    python lpex2.py <filename> b     barrier
#    python lpex2.py <filename> h     barrier with crossover
#    python lpex2.py <filename> s     sifting
#    python lpex2.py <filename> c     concurrent



from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexSolverError
import numpy as np


DEBUG=0


def lpex2(filename, method):
    c = cplex.Cplex(filename)

    alg = c.parameters.lpmethod.values

    if method == "o":
        c.parameters.lpmethod.set(alg.auto)
    elif method == "p":
        c.parameters.lpmethod.set(alg.primal)
    elif method == "d":
        c.parameters.lpmethod.set(alg.dual)
    elif method == "b":
        c.parameters.lpmethod.set(alg.barrier)
        c.parameters.barrier.crossover.set(
            c.parameters.barrier.crossover.values.none)
    elif method == "h":
        c.parameters.lpmethod.set(alg.barrier)
    elif method == "s":
        c.parameters.lpmethod.set(alg.sifting)
    elif method == "c":
        c.parameters.lpmethod.set(alg.concurrent)
    else:
        raise ValueError(
            'method must be one of "o", "p", "d", "b", "h", "s" or "c"')

    try:        
        c.set_log_stream(None)
        c.set_error_stream(None)
        c.set_warning_stream(None)
        c.set_results_stream(None)
        c.solve()
    except CplexSolverError:
        print("Exception raised during solve")
        return

    # # solution.get_status() returns an integer code
    # status = c.solution.get_status()
    # print(c.solution.status[status])
    # if status == c.solution.status.unbounded:
    #     print("Model is unbounded")
    #     return
    # if status == c.solution.status.infeasible:
    #     print("Model is infeasible")
    #     return
    # if status == c.solution.status.infeasible_or_unbounded:
    #     print("Model is infeasible or unbounded")
    #     return

    # s_method = c.solution.get_method()
    # s_type = c.solution.get_solution_type()

    # if DEBUG:
    # 	print("Solution status = ", status, ":", end=' ')
	   #  # the following line prints the status as a string
	   #  print(c.solution.status[status])
	   #  print("Solution method = ", s_method, ":", end=' ')
	   #  print(c.solution.method[s_method])

    # if s_type == c.solution.type.none:
    #     print("No solution available")
    #     return
    # if DEBUG:
    # 	print("Objective value = ", c.solution.get_objective_value())

    # if s_type == c.solution.type.basic:
    #     basis = c.solution.basis.get_col_basis()
    # else:
    #     basis = None

    # print()

    # if DEBUG:
    # 	x = c.solution.get_values(0, c.variables.get_num() - 1)
	   #  # because we're querying the entire solution vector,
	   #  # x = c.solution.get_values()
	   #  # would have the same effect
	   #  for j in range(c.variables.get_num()):
	   #      print("Column %d: Value = %17.10g" % (j, x[j]))
	   #      if basis is not None:
	   #          if basis[j] == c.solution.basis.status.at_lower_bound:
	   #              print("  Nonbasic at lower bound")
	   #          elif basis[j] == c.solution.basis.status.basic:
	   #              print("  Basic")
	   #          elif basis[j] == c.solution.basis.status.at_upper_bound:
	   #              print("  Nonbasic at upper bound")
	   #          elif basis[j] == c.solution.basis.status.free_nonbasic:
	   #              print("  Superbasic, or free variable at zero")
	   #          else:
	   #              print("  Bad basis status")


	   #  infeas = c.solution.get_float_quality(
	   #      c.solution.quality_metric.max_primal_infeasibility)
	   #  print("Maximum bound violation = ", infeas)



    # multipliers_list=[]
    # # print c.solution.get_values()
    # # print c.variables.get_names()
    # for i, x in enumerate(c.variables.get_names()):
    #     if(x.startswith('a')):
    #         multipliers_list.append(c.solution.get_values(i))

    # return np.array(multipliers_list)

    # return np.array(c.solution.get_values())
    return c




