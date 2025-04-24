import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os

# Set up working directory
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results")

# Load data
data_matrix = pd.read_csv("Data1.csv", header=None).values
weight_matrix = pd.read_csv("Weight1.csv", header=None).values

n_set = range(4)
m_set = range(10)

# Initialize model
model = pyo.ConcreteModel()

# Variables
model.d_plus = pyo.Var(m_set, bounds=(0, None))
model.d_minus = pyo.Var(m_set, bounds=(0, None))

def weight_expr(model, j):
    return weight_matrix[j, 0] + model.d_plus[j] - model.d_minus[j]

model.w_ij = pyo.Var(n_set, m_set)

def weighted_value_rule(model, i, j):
    return model.w_ij[i, j] == data_matrix[i, j] * weight_expr(model, j)

model.weighted_value_constraint = pyo.Constraint(n_set, m_set, rule=weighted_value_rule)

# Calculate PIS and NIS directly from the weighted values
model.PIS = pyo.Var(m_set, bounds=(0, None))
model.NIS = pyo.Var(m_set, bounds=(0, None))


model.pis_constraints = pyo.ConstraintList()
model.nis_constraints = pyo.ConstraintList()
for j in m_set:
    for i in n_set:
        model.pis_constraints.add(model.PIS[j] >= model.w_ij[i, j])
        model.nis_constraints.add(model.NIS[j] <= model.w_ij[i, j])
        

    
# Distance and TOPSIS score (nonlinear expressions still present here)
model.Di_plus = pyo.Var(n_set, bounds=(0, None))
model.Di_minus = pyo.Var(n_set, bounds=(0, None))

def d_plus_rule(model, i):
    return model.Di_plus[i] == pyo.sqrt(sum((model.w_ij[i, j] - model.PIS[j])**2 for j in m_set))

def d_minus_rule(model, i):
    return model.Di_minus[i] == pyo.sqrt(sum((model.w_ij[i, j] - model.NIS[j])**2 for j in m_set))

model.d_plus_constraint = pyo.Constraint(n_set, rule=d_plus_rule)
model.d_minus_constraint = pyo.Constraint(n_set, rule=d_minus_rule)

model.Ci = pyo.Var(n_set, bounds=(0, 1))

def topsis_score_rule(model, i):
    return model.Ci[i] == model.Di_minus[i] / (model.Di_plus[i] + model.Di_minus[i])

model.topsis_score_constraint = pyo.Constraint(n_set, rule=topsis_score_rule)

# Ensure alternative 2 ranks first
eps = 0.00001
model.rank_constraints = pyo.ConstraintList()
for i in n_set:
    if i != 1:
        model.rank_constraints.add(model.Ci[1] >= model.Ci[i] + eps)

# Weight constraints
model.weight_sum = pyo.Constraint(expr=sum(weight_expr(model, j) for j in m_set) == 1)
model.weight_upper_bound = pyo.Constraint(m_set, rule=lambda model, j: weight_expr(model, j) <= 1)
model.weight_lower_bound = pyo.Constraint(m_set, rule=lambda model, j: weight_expr(model, j) >= 0)
model.weight_balance = pyo.Constraint(expr=sum(model.d_plus[j] - model.d_minus[j] for j in m_set) == 0)

# Objective
model.Cost = pyo.Objective(expr=sum(model.d_plus[j] + model.d_minus[j] for j in m_set), sense=pyo.minimize)

# Solve with IPOPT (supports nonlinear constraints)
opt = pyo.SolverFactory("ipopt")
result = opt.solve(model, tee=True)

 

# Output solver status and termination condition
print("Solver status:", result.solver.status)
print("Solver termination condition:", result.solver.termination_condition)


# Display model results
model.display()

# Extract final results
results = np.zeros((1, 4))
for i in n_set:
    results[0, i] = pyo.value(model.Ci[i])

print("\nFinal TOPSIS Scores:")
print(results)

# Print individual scores
for i in n_set:
    print(f"Ci[{i}] = {pyo.value(model.Ci[i])}")

# Save results to an Excel file
df_results = pd.DataFrame(results)
df_results.to_excel("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/TOPSIS_results.xlsx", sheet_name="Results", index=False, header=False)

# Print weight adjustments
for j in m_set:
    print(f"d_plus[{j}] = {pyo.value(model.d_plus[j])}")
    print(f"d_minus[{j}] = {pyo.value(model.d_minus[j])}")

# Save d_plus and d_minus values to an Excel file
data_minus = {'Variable': [], 'Value': []}
data_plus = {'Variable': [], 'Value': []}

for j in m_set:
    data_minus['Variable'].append(f'd_minus[{j}]')
    data_minus['Value'].append(pyo.value(model.d_minus[j]))

    data_plus['Variable'].append(f'd_plus[{j}]')
    data_plus['Value'].append(pyo.value(model.d_plus[j]))

df_minus = pd.DataFrame(data_minus)
df_plus = pd.DataFrame(data_plus)

adjusted_weights = np.zeros(len(m_set))
for j in m_set:
    adjusted_weights[j] = weight_matrix[j, 0] + pyo.value(model.d_plus[j]) - pyo.value(model.d_minus[j])
print("Adjusted Weights:\n", adjusted_weights)


# Define full path to your desired folder
folder_path = "/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results"
file_path = f"{folder_path}/weight2.csv"
np.savetxt(file_path, adjusted_weights, delimiter=",", fmt="%.10f")

with pd.ExcelWriter('Variable_TOPSIS.xlsx') as writer:
    df_minus.to_excel(writer, sheet_name='d_minus', index=False)
    df_plus.to_excel(writer, sheet_name='d_plus', index=False)

# 2) The weighted matrix w_ij
for i in n_set:
    print([pyo.value(model.w_ij[i,j]) for j in m_set])

print("\nWeighted Matrix w_ij:")
for i in n_set:
    print([round(pyo.value(model.w_ij[i,j]), 4) for j in m_set])