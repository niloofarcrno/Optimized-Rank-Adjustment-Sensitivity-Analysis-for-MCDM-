# Optimized-Rank-Adjustment-Decision Making Robustness Analysis
Example: The non-linear and linear mathematical models are developed to perform sensitivity analysis within the ANP and AHP frameworks. This tool determines the minimum criteria weight changes needed to make the second alternative the top-ranked option. It aids in evaluating the robustness of the rankings and the method used in a specific case study.

Upon solving the model, the optimal decision variables—representing the changes within each criterion—are obtained. This approach significantly reduces the time required for sensitivity analysis, especially for cases with a large number of criteria. The graph below illustrates the results for the ANP and AHP sensitivity analysis models.
<img width="852" alt="image" src="https://github.com/niloofarcrno/Optimized-Rank-Adjustment-Sensitivity-Analysis/assets/141967064/bc50ab0f-bfdf-4bc7-8e47-2ed0ba54e22a">




The tool developed for the ANP model is open access, leveraging the open-source solver IPOPT.

Also, optimization models (NLPSensitivity_TOPSIS.py) is developed to determine the minimal weight adjustments required to make the second alternative rank first in the TOPSIS method.

Implements a nonlinear optimization model using Pyomo and IPOPT.

Ensures calculated TOPSIS scores (Ci) reflect dynamic weight changes.

Objective: Minimize the sum of d_plus and d_minus (weight modifications).

Constraints ensure valid weight normalization, upper/lower bounds, and that the second alternative ranks highest.

Final adjusted weights are exported to weight2.csv (no headers).

Validates output against static TOPSIS results.
