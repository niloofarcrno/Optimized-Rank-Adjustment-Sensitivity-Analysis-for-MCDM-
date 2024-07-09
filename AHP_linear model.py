#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:17:36 2024

@author: niloofarakbarian
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import os

# Setup
os.environ['PATH'] += os.pathsep + '/opt/homebrew/bin'
print("Updated PATH:", os.environ['PATH'])

os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results")

# Load data and weight matrices
data_matrix = pd.read_csv("Data.csv", header=None).values  # Assuming 4x11 matrix
weight_matrix = pd.read_csv("Weight.csv", header=None).values  # Assuming 11x4 matrix

# Ensure the matrices have correct dimensions
assert data_matrix.shape == (4, 11), "Data matrix should be 4x11"
assert weight_matrix.shape == (11, 4), "Weight matrix should be 11x4"

# Define sets
n_set = range(4)  # 4 alternatives
m_set = range(11)  # 11 criteria

model = pyo.ConcreteModel()

# Define parameters and variables

model.d_minus = pyo.Var(m_set, n_set, bounds=(0, None), initialize=0)
model.d_plus = pyo.Var(m_set, n_set, bounds=(0, None), initialize=0)
model.results = pyo.Var(n_set, initialize=0)

# Define the weight expression
def weight_expr(model, i, j):
    return weight_matrix[j, i] + model.d_plus[j, i] - model.d_minus[j, i]

# Define the sum product expression
def results_rule(model, i):
    return model.results[i] == sum(data_matrix[i, j] * weight_expr(model, i, j) for j in m_set)

model.results_constraint = pyo.Constraint(n_set, rule=results_rule)

# Objective function (dummy objective, since we are only interested in constraints)
model.Cost = pyo.Objective(expr=sum(model.d_plus[j, i] + model.d_minus[j, i] for j in m_set for i in n_set), sense=pyo.minimize)

# Balance constraints
model.balance1 = pyo.Constraint(n_set, m_set, rule=lambda model, i, j: weight_expr(model, i, j) <= 1)
model.balance2 = pyo.Constraint(n_set, m_set, rule=lambda model, i, j: weight_expr(model, i, j) >= 0)
model.balance4 = pyo.Constraint(n_set, m_set, rule=lambda model, i, j: model.d_plus[j, i] >= 0)
model.balance5 = pyo.Constraint(n_set, m_set, rule=lambda model, i, j: model.d_minus[j, i] >= 0)
model.balance6 = pyo.Constraint(n_set, rule=lambda model, i: sum(model.d_plus[j, i] - model.d_minus[j, i] for j in m_set) == 0)

# Add constraints to ensure the first component is greater than the others
model.rank_constraints = pyo.ConstraintList()
for i in n_set:
    if i != 0:
        model.rank_constraints.add(model.results[0] >= model.results[i])

model.balance8 = pyo.Constraint(rule=lambda model: (
    sum(data_matrix[0, j] * weight_expr(model, 0, j) for j in m_set) >=
    sum(data_matrix[1, j] * weight_expr(model, 1, j) for j in m_set)
))

model.balance9 = pyo.Constraint(rule=lambda model: (
    sum(data_matrix[1, j] * weight_expr(model, 1, j) for j in m_set) >=
    sum(data_matrix[2, j] * weight_expr(model, 2, j) for j in m_set)
))

model.balance10 = pyo.Constraint(rule=lambda model: (
    sum(data_matrix[1, j] * weight_expr(model, 1, j) for j in m_set) >=
    sum(data_matrix[3, j] * weight_expr(model, 3, j) for j in m_set)
))

# Solve the model using Gurobi
opt = SolverFactory("gurobi")
result = opt.solve(model, tee=True)

# Check solver status
print("Solver status:", result.solver.status)
print("Solver termination condition:", result.solver.termination_condition)

# Display the results
model.display()

# Extract the final results values
results = np.zeros((1, 4))
for i in n_set:
    results[0, i] = pyo.value(model.results[i])

print("\nResults Matrix (Final):")
print(results)

# Print the results
for i in n_set:
    print(f"results[{i}] = {pyo.value(model.results[i])}")

# Save results to an Excel file
df_results = pd.DataFrame(results)
df_results.to_excel("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/results.xlsx", sheet_name="Results", index=False, header=False)

# Print the solution for d_plus and d_minus
for i in n_set:
    for j in m_set:
        print(f"d_plus[{j, i}] = {pyo.value(model.d_plus[j, i])}")
        print(f"d_minus[{j, i}] = {pyo.value(model.d_minus[j, i])}")

# Create DataFrames to store the variables
data_minus = {'Variable': [], 'Value': []}
data_plus = {'Variable': [], 'Value': []}

for i in n_set:
    for j in m_set:
        data_minus['Variable'].append(f'd_minus{j, i}]')
        data_minus['Value'].append(pyo.value(model.d_minus[j, i]))

for i in n_set:
    for j in m_set:
        data_plus['Variable'].append(f'd_plus{j, i}]')
        data_plus['Value'].append(pyo.value(model.d_plus[j, i]))

# Create DataFrames
df_minus = pd.DataFrame(data_minus)
df_plus = pd.DataFrame(data_plus)

# Save each DataFrame to a separate sheet in an Excel file
with pd.ExcelWriter('Variable_AHP.xlsx') as writer:
    df_minus.to_excel(writer, sheet_name='C')
    df_plus.to_excel(writer, sheet_name='D')
