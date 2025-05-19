battery_model_sensitivity_analysis
==============================

A comprehensive framework for global sensitivity analysis of lithium battery models, supporting multiple modeling approaches and sensitivity methods to identify key parameters and improve model accuracy.

## Overview

This project aims to analyze the sensitivity of geometric parameters in the Single Particle Model with electrolyte (SPMe) for lithium-ion batteries using the Sobol method. By identifying key parameters, this work can better guide battery design, optimization, and manufacturing processes. The detailed analysis and code can be found in the Jupyter Notebook: [notebooks/spme_geometric_sobol_analysis.ipynb](./notebooks/spme_geometric_sobol_analysis.ipynb).

## Core Libraries

-   **PyBaMM (Python Battery Mathematical Modelling)**: An open-source Python package for building and simulating lithium-ion battery models.
-   **SALib (Sensitivity Analysis Library in Python)**: A Python library for sensitivity analysis, providing various methods including Sobol.

## Method

To quantify the impact of parameter variations on the model output (time-series data, such as voltage curves), we define the following difference metrics:

1.  **Positional Difference ($\delta_p$)**: Measures the overall positional shift.
    $$\delta_p[Y(t), Y_b(t)] = \text{mean}[Y(t)] - \text{mean}[Y_b(t)]$$
2.  **Scale Difference ($\delta_s$)**: Measures the difference in amplitude.
    $$\delta_s[Y(t), Y_b(t)] = (\max[Y(t)] - \min[Y(t)]) - (\max[Y_b(t)] - \min[Y_b(t)])$$
3.  **Shape Difference ($\delta_r$)**: Measures the shape difference after correcting for position and scale, calculated using Root Mean Square Error (RMSE).
    $$\delta_r[Y(t), Y_b(t)] = \text{RMSE}(Y(t), Y_b(t)) = \sqrt{\frac{1}{M} \sum_{i=1}^{M} (Y(t_i) - Y_b(t_i))^2}$$

This project uses a composite difference metric $\Delta_i = \delta_p + \delta_s + \delta_r$ as the target function output for the Sobol analysis.

## Example Results

The following images show partial results from the Sobol sensitivity analysis of SPMe model geometric parameters. The full analysis can be found in [notebooks/spme_geometric_sobol_analysis.ipynb](./notebooks/spme_geometric_sobol_analysis.ipynb).

**S1 and ST Indices (based on $\Delta_i$)**

![Sobol S1 and ST Indices](./results/sobol_s1_st_indices_delta_i.png)

This chart displays the first-order sensitivity indices (S1) and total sensitivity indices (ST) for each parameter. S1 represents the contribution of a single parameter to the variance of the model output, while ST represents the total contribution of the parameter itself and its interactions with other parameters to the model output variance.

**S2 Indices Heatmap (based on $\Delta_i$)**

![Sobol S2 Heatmap](./results/sobol_s2_heatmap_delta_i.png)

This heatmap shows the second-order sensitivity indices (S2) between parameters, measuring the impact of interactions between pairs of parameters on the model output.

## Results Interpretation

Based on the Sobol analysis results (see [notebooks/spme_geometric_sobol_analysis.ipynb](./notebooks/spme_geometric_sobol_analysis.ipynb) for details):

1.  **Key Parameter Sensitivities:**
    *   Some parameters (e.g., `Positive electrode thickness [m]`, `Negative particle radius [m]`) exhibit high S1 and ST values, indicating they have the most significant impact on the model output.
    *   Other parameters show moderate or low sensitivity.

2.  **Parameter Interaction Effects:**
    *   When a parameter's ST index is significantly higher than its S1 index, or the S2 heatmap shows strong S2 values between specific parameter pairs, it indicates significant interaction effects between these parameters.

## Future Work

This analysis can be further improved in several ways:
- Increase the number of parameters analyzed.
- Refine the sampling range of parameters.
- Increase the number of samples to improve the robustness of the results.
- Optimize the difference quantification method, for example, by considering weighting or normalization of the difference metrics.



