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
super_matrix=pd.read_csv("Supermatrix.csv", header=None)

#the supermatrix 15*15 dimension 
n_set=range(15)
m_set=range(15)

model = pyo.ConcreteModel()

# Define parameters and variables


model.d_plus = pyo.Var(range(11), range(11, 15), initialize=0)
model.d_minus = pyo.Var(range(11), range(11, 15), initialize=0)
model.limitmatrix = pyo.Var(n_set,m_set, initialize=0)
model.limitmatrix2 = pyo.Var(n_set,m_set, initialize=0)
model.limitmatrix3 = pyo.Var(n_set,m_set, initialize=0)

#apply changes only if the row and columsn are within a requested range 
def weight_expr(model, i, j):
    # Apply changes only for columns 11-14 and rows 0-10
    if 0 <= i <= 10 and 11 <= j <= 14:
        return super_matrix.iloc[i, j] + model.d_plus[i, j] - model.d_minus[i, j]
    else:
        # Return the original value from the supermatrix if out of the specified range
        return super_matrix.iloc[i, j]



# Combine the iterative results to approximate the final results x^2
def results_rule(model, i,j):
    return model.limitmatrix[i,j] == sum(weight_expr(model, i, k)* weight_expr(model, k, j) for k in m_set)

#power 3: x^3
def limit_rule(model, i,j):
    return model.limitmatrix2[i,j] == sum(model.limitmatrix[i,k]* weight_expr(model, k, j) for k in m_set)

#power 5:x^5
def limit2_rule(model, i,j):
    return model.limitmatrix3[i,j] == sum(model.limitmatrix2[i,k]* model.limitmatrix[k,j] for k in m_set)

model.limitmatrix3_constraint = pyo.Constraint(n_set, m_set, rule=limit2_rule)

model.limitmatrix2_constraint = pyo.Constraint(n_set, m_set, rule=limit_rule)

model.limitmatrix_constraint = pyo.Constraint(n_set,m_set, rule=results_rule)

# Objective function (dummy objective, since we are only interested in constraints)
model.Cost = pyo.Objective(expr=sum(model.d_plus[i, j] + model.d_minus[i, j] for j in range(11,15) for i in range(11)), sense=pyo.minimize)

# Balance constraints
model.balance1 = pyo.Constraint(n_set, m_set, rule=lambda model, i, j: model.limitmatrix3[i,j] <= 1)
model.balance2 = pyo.Constraint(n_set, m_set, rule=lambda model, i, j: model.limitmatrix3[i,j] >= 0)
model.balance4 = pyo.Constraint(range(11), range(11, 15), rule=lambda model, i, j: model.d_plus[i, j] >= 0)
model.balance5 = pyo.Constraint(range(11), range(11, 15), rule=lambda model, i, j: model.d_minus[i, j] >= 0)
model.balance6 = pyo.Constraint(range(11,15), rule=lambda model, j: sum(model.d_plus[i, j] - model.d_minus[i, j] for i in range(11)) == 0)


model.balance8 = pyo.Constraint(m_set,rule=lambda model,j: (
   model.limitmatrix3[11,j] >=
   model.limitmatrix3[12,j]
))


model.balance9 = pyo.Constraint(m_set,rule=lambda model,j: (
   model.limitmatrix3[11,j] >=
   model.limitmatrix3[13,j]
))

model.balance10 = pyo.Constraint(m_set,rule=lambda model,j: (
   model.limitmatrix3[11,j] >=
  model.limitmatrix3[14,j]
))


# Solve the model using Gurobi
opt = SolverFactory("ipopt")
result = opt.solve(model, tee=True)

# Check solver status
print("Solver status:", result.solver.status)
print("Solver termination condition:", result.solver.termination_condition)

# Display the results
model.display()

        
# Extract the final results values
results = np.zeros((15, 15))
for i in n_set:
    for j in m_set:
        results[i, j] = pyo.value(model.limitmatrix3[i,j])

print("\nResults Matrix (Final):")
print(results)

# # Print the results
# for i in n_set:
#     for j in m_set:
#         print(f"results[{i,j}] = {pyo.value(model.limitmatrix2[i,j])}")

# Save results to an Excel file
df_results = pd.DataFrame(results)
df_results.to_excel("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/results.xlsx", sheet_name="Results", index=False, header=False)

# Print the solution for d_plus and d_minus
for i in range (11):
    for j in range(11,15):
        print(f"d_plus[{i, j} = {pyo.value(model.d_plus[i, j])}")
        print(f"d_minus[{i, j}] = {pyo.value(model.d_minus[i, j])}")

# Create DataFrames to store the variables
data_minus = {'Variable': [], 'Value': []}
data_plus = {'Variable': [], 'Value': []}

for i in range (11):
    for j in range(11,15):
        data_minus['Variable'].append(f'd_minus{i, j}]')
        data_minus['Value'].append(pyo.value(model.d_minus[i, j]))

for i in range (11):
    for j in range(11,15):
        data_plus['Variable'].append(f'd_plus{i, j}]')
        data_plus['Value'].append(pyo.value(model.d_plus[i, j]))

# Create DataFrames
df_minus = pd.DataFrame(data_minus)
df_plus = pd.DataFrame(data_plus)

# Save each DataFrame to a separate sheet in an Excel file
with pd.ExcelWriter('Variable.xlsx') as writer:
    df_minus.to_excel(writer, sheet_name='C')
    df_plus.to_excel(writer, sheet_name='D')
    
    
    
    
    
    
