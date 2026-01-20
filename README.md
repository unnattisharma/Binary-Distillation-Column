# Binary Distillation Column Simulator
**First-Principles Steady-State Model (CMO)**

## 1. Purpose

This package contains a from-scratch implementation of a steady-state binary distillation column model developed using first-principles mass, equilibrium, summation, and energy (MESH) equations under Constant Molar Overflow (CMO) assumptions. The working system modeled is benzene–toluene.

## 2. Contents of the ZIP File

After unzipping, the folder should contain:

```text
thermodynamics.py          # Antoine equation, bubble point, dew point, mixture enthalpy
solver.py                  # Newton–Raphson solver, Jacobian construction
binary_dist_column_CMO.py  # Column MESH equations, flow calculations, column profile
analysis.py                # Sensitivity, optimization, equipment/operational diagnostics
model.py                   # Main execution script
requirements.txt           # Python dependencies
generated_figures/         # Auto-generated plots
README.md                  # This file
```

No additional data files are required.

## 3. Software Requirements

**Recommended:**

  * Python version: 3.12.12
  * Required Python packages:
    * NumPy (2.0.2)
    * SciPy (1.16.3)
    * Matplotlib (3.10.0)

**Minimum required:**

  * Python version: 3.9 or newer
  * Required Python packages:
    * NumPy (1.23.0)
    * SciPy (1.9.0)
    * Matplotlib (3.6.0)

## 4. Installation Instructions

1. Unzip the files to any local directory.

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## 5. Running the Model

From the unzipped directory, run:

```bash
python model.py
```

This script:

* Generates the VLE diagram
* Solves the steady-state distillation column
* Performs degradation, sensitivity, optimization, and upset condition analyses
* Displays and saves plots automatically

All figures are saved to the folder `generated_figures/` with timestamps.

## 6. Key Outputs

The model computes and reports:

* Distillate benzene purity (xD)
* Bottoms toluene purity (1-xB)
* Condenser heat duty (QC)
* Reboiler heat duty (QB)
* Tray temperature and composition profiles
* Sensitivity trends with respect to:

  * Reflux ratio
  * Feed stage location
  * Feed quality (q)
* Operating envelope and disturbance response

## 7. Modeling Assumptions

* Binary system (benzene–toluene)
* Steady-state operation
* Constant molar overflow
* Ideal VLE (Raoult’s law)
* No pressure drop
* Murphree vapor-phase tray efficiency