#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:39:52 2025

@author: niloofarakbarian
"""
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import os

# Set up the environment and working directory
os.environ['PATH'] += os.pathsep + '/opt/homebrew/bin'
print("Updated PATH:", os.environ['PATH'])

# ------------------------------
# STEP 1: Load normalized matrix and weights from CSV
# ------------------------------

# Change to the appropriate working directory
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results")

# Load normalized data and weight matrices
data_matrix = pd.read_csv("Data1.csv", header=None).values  # Expecting a 4x10 normalized matrix
weight_matrix = pd.read_csv("Weight1.csv", header=None).values
weights = weight_matrix[:, 0]  # Use only the first column

weighted_matrix = data_matrix * weights  # Broadcasting works now


# ------------------------------
# STEP 3: Identify PIS (best) and NIS (worst)
# ------------------------------
PIS = np.max(weighted_matrix, axis=0)
NIS = np.min(weighted_matrix, axis=0)

# ------------------------------
# STEP 4: Calculate distances to PIS and NIS
# ------------------------------
D_plus = np.sqrt(np.sum((weighted_matrix - PIS)**2, axis=1))
D_minus = np.sqrt(np.sum((weighted_matrix - NIS)**2, axis=1))

# ------------------------------
# STEP 5: Calculate relative closeness to ideal solution
# ------------------------------
Ci = D_minus / (D_plus + D_minus)

# ------------------------------
# STEP 6: Rank the alternatives
# ------------------------------
rank = (-Ci).argsort() + 1  # Rank 1 = best

# ------------------------------
# Output results
# ------------------------------
results_df = pd.DataFrame({
    'Alternative': [f"A{i+1}" for i in range(len(Ci))],
    'D+ (PIS Distance)': D_plus,
    'D- (NIS Distance)': D_minus,
    'TOPSIS Score (Ci)': Ci,
    'Rank': rank
})

print(results_df.sort_values('Rank'))

# Optional: save to Excel
results_df.to_excel("topsis_results.xlsx", index=False)