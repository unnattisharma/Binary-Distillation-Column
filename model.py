# model.py
"""
Main script combining execution of all the thermodynamics, physical principles, numerical methods, visualization and analysis.
"""
from binary_dist_column_CMO import Column
import numpy as np
from analysis import EquipmentDeterioration, ProcessOptimization, UpsetConditions

# First-Principles Model Development
distillation_column = Column() #initialize model
VLE_plot = distillation_column.thermo.generate_VLE()

column_results = distillation_column.run_distillation(display_output='yes') #solve for column profiles

# Scenario A
column_deterioration_analysis = EquipmentDeterioration()
degradation_results = column_deterioration_analysis.degraded_efficiency_case(distillation_column,1)
deterioration_results=column_deterioration_analysis.diagnose_equipment_issues(distillation_column)

# Scenario B
column_optimization_analysis = ProcessOptimization()
R_sweep = column_optimization_analysis.reflux_ratio_sensitivity_analysis(distillation_column)
fs_sweep = column_optimization_analysis.feed_stage_sensitivity_analysis(distillation_column)
q_sweep = column_optimization_analysis.feed_condition_sensitivity_analysis(distillation_column)
optimization_results=column_optimization_analysis.process_optimization(distillation_column)

# Scenario C
column_upset_analysis = UpsetConditions()
disturbance_results = column_upset_analysis.process_disturbances(distillation_column,case_number=3)
operating_envelope_results=column_upset_analysis.find_operating_envelope(distillation_column)