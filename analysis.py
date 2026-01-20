# analysis.py
"""
Custom functions for Data generation and scenario analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
from binary_dist_column_CMO import Column
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator
import os
from datetime import datetime

# prepare to sav egenrated figures
SAVE_DIR = 'generated_figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class EquipmentDeterioration:

  def degraded_efficiency_case(self,column,case_number,tray_efficiency_M_baseline=None,tray_efficiency_M_test=None,N_stages=None):
    """Compare column profiles against baseline when Murphree efficiency is varied. Use case number 1-3 for predefined scenarios, and case number 4 for custom degradation scenario."""

    if tray_efficiency_M_baseline is None: tray_efficiency_M_baseline = column.tray_efficiency_M
    if N_stages is None: N_stages = column.N_stages

    fig, (ax14, ax15, ax16) = plt.subplots(1, 3)
    fig.suptitle('Column Profiles')
    ax14.set_title('Temperature')
    ax15.set_title('y_Benzene')
    ax16.set_title('x_Toluene')
    ax14.set_xlabel('Tray #')
    ax15.set_xlabel('Tray #')
    ax16.set_xlabel('Tray #')
    ax14.set_ylabel('temperature (C)')
    ax15.set_ylabel('mole fraction')
    ax16.set_ylabel('mole fraction')
    fig.set_size_inches(12,4)

    # baseline calculation
    [outputs, Tn, y_real, xn] = column.run_distillation(tray_efficiency_M=tray_efficiency_M_baseline,N_stages=N_stages,display_output='no')
    
    ax14.plot(Tn, 'k', label = 'baseline')
    ax15.plot(y_real, 'k')
    ax16.plot(1-xn, 'k')

    if case_number == 1:
      tray_efficiency_M_test = 0.6*np.ones(N_stages)
      print("\n--DEGRADED CASE 1: UNIFORM REDUCTION--")
    if case_number == 2:
      tray_efficiency_M_test = 0.7*np.ones(N_stages)
      for i in range(4,8):
        tray_efficiency_M_test[i] = 0.5
      print("\n--DEGRADED CASE 2: LOCALIZED DAMAGE--")
    if case_number == 3:
      tray_efficiency_M_test = np.linspace(0.7, 0.55, 15)
      print("\n--DEGRADED CASE 3: PROGRESSIVE DEGRADATION--")
      print("\nMurphree efficiencies for trays 1-15:", tray_efficiency_M_test)
    if case_number == 4:
      if tray_efficiency_M_test is None: tray_efficiency_M_test = tray_efficiency_M_baseline
      print("\n--DEGRADED CASE 4: CUSTOM DEGRADATION--")
      print("\nMurphree efficiencies for trays 1-15:", tray_efficiency_M_test)
    
    # detemrine column profile for degraded/test efficiencies
    [outputs_test, Tn_test, y_real_test, xn_test] = column.run_distillation(tray_efficiency_M=tray_efficiency_M_test,N_stages=N_stages,display_output='no')
    
    ax14.plot(Tn_test, label = 'case '+str(case_number))
    ax15.plot(y_real_test)
    ax16.plot(1-xn_test)

    fig.legend(loc = 'right')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(SAVE_DIR, f'degradation_column_profiles_case{case_number}_{timestamp}.png'))
    plt.show()
    return [outputs_test, Tn_test, y_real_test, xn_test]

  def diagnose_equipment_issues(self,column,tray_efficiency_M_baseline=None,tray_efficiency_M_test=None,N_stages=None):
    """Visualize impact on column performance for all efficiency degradation cases."""

    if tray_efficiency_M_baseline is None: tray_efficiency_M_baseline = column.tray_efficiency_M
    if N_stages is None: N_stages = column.N_stages

    results = []
    results.append(column.run_distillation(tray_efficiency_M=tray_efficiency_M_baseline,N_stages=N_stages,display_output='no'))
    results.append(self.degraded_efficiency_case(column,1,tray_efficiency_M_baseline=tray_efficiency_M_baseline))
    results.append(self.degraded_efficiency_case(column,2,tray_efficiency_M_baseline=tray_efficiency_M_baseline))
    results.append(self.degraded_efficiency_case(column,3,tray_efficiency_M_baseline=tray_efficiency_M_baseline))

    if tray_efficiency_M_test is not None:
      results.append(self.degraded_efficiency_case(column,4,tray_efficiency_M_baseline=tray_efficiency_M_baseline,tray_efficiency_M_test=tray_efficiency_M_test))

    xD = []
    xB_tol = []
    Qc = []
    Qb = []

    for i in range(len(results)):
      xD.append(results[i][0]['Benzene purity in distillate (xD)'])
      xB_tol.append(results[i][0]['Toluene purity in bottoms (1-xB)'])
      Qc.append(results[i][0]['Condenser heat duty (QC) MW'])
      Qb.append(results[i][0]['Reboiler heat duty (QB) MW'])

    fig, (ax6, ax7, ax8) = plt.subplots(1, 3)
    fig.suptitle('Equipment degradation & performance')
    fig.set_size_inches(12,4)

    ax6.plot(xD, label = 'xD_benzene')
    ax6.set_title('Distillate purity')
    ax6.set_xlabel('degradation case')
    ax6.set_ylabel('mole fraction')

    ax7.plot(xB_tol, label = 'xB_toluene')
    ax7.set_title('Bottoms purity')
    ax7.set_xlabel('degradation case')
    ax7.set_ylabel('mole fraction')

    ax8.plot(abs(np.array(Qc)), label = '|QC|')
    ax8.plot(Qb, label = 'QB')
    ax8.plot(np.array(Qb)+abs(np.array(Qc)), label = 'QB + |QC|')
    ax8.set_title('Heat Duty')
    ax8.set_xlabel('degradation case')
    ax8.set_ylabel('energy (MW)')

    fig.legend(loc = 'right')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(SAVE_DIR, f'degradation_vs_performance_{timestamp}.png'))
    plt.show()
    return

class ProcessOptimization:

  def reflux_ratio_sensitivity_analysis(self,column,R_values=np.linspace(1.5,4.0,6)):
    """Observe performance sensitivity to reflux ratio R. R_values can take custom set of test reflux ratios."""

    xD_values = []
    xB_tol_values = []
    Qc_values = []
    Qb_values = []

    for R in R_values:
      outputs = column.run_distillation(R=R,display_output='no')[0]
      xD_values.append(outputs['Benzene purity in distillate (xD)'])
      xB_tol_values.append(outputs['Toluene purity in bottoms (1-xB)'])
      Qc_values.append(outputs['Condenser heat duty (QC) MW'])
      Qb_values.append(outputs['Reboiler heat duty (QB) MW'])

    fig, ((ax6, ax7), (ax8, ax9)) = plt.subplots(2, 2)
    fig.suptitle('Reflux Ratio Sensitivity Analysis')
    fig.set_size_inches(10,10)

    ax6.plot(R_values, xD_values, 'bo', label = 'xD_benzene')
    ax6.plot(R_values, xB_tol_values, 'r', label = 'xB_toluene')
    ax6.set_title('Product purity')
    ax6.set_xlabel('R')
    ax6.set_ylabel('mole fraction')

    ax7.plot(R_values, abs(np.array(Qc_values)), 'go', label = '|QC|')
    ax7.plot(R_values, Qb_values, 'm--', label = 'QB')
    ax7.plot(R_values, np.array(Qb_values)+abs(np.array(Qc_values)), 'k', label = 'QB + |QC|')
    ax7.set_title('Heat Duty')
    ax7.set_xlabel('R')
    ax7.set_ylabel('energy (MW)')

    ax8.plot(R_values, np.array(Qb_values)+np.array(Qc_values), 'k.', label = 'QB + QC')
    ax8.set_title('Process Net Energy Consumption')
    ax8.set_xlabel('R')
    ax8.set_ylabel('energy (MW)')

    ax9.plot(np.array(Qb_values)+abs(np.array(Qc_values)), xD_values, 'bo')
    ax9.plot(np.array(Qb_values)+abs(np.array(Qc_values)), xB_tol_values, 'r')
    ax9.set_title('Purity vs Energy')
    ax9.set_xlabel('Total Heat Duty (MW)')
    ax9.set_ylabel('mole fraction')
    
    fig.legend(loc = 'right')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(SAVE_DIR, f'R_sensitivity_{timestamp}.png'))
    plt.show()
    return

  def feed_stage_sensitivity_analysis(self,column,feed_stage_values=np.linspace(5,11,7)):
    """Observe performance sensitivity to the feed stage. feed_stage_values can take custom set of test feed stage."""

    xD_values = []
    xB_tol_values = []
    Qc_values = []
    Qb_values = []

    for fs in feed_stage_values:
      outputs = column.run_distillation(feed_stage=fs,display_output='no')[0]
      xD_values.append(outputs['Benzene purity in distillate (xD)'])
      xB_tol_values.append(outputs['Toluene purity in bottoms (1-xB)'])
      Qc_values.append(outputs['Condenser heat duty (QC) MW'])
      Qb_values.append(outputs['Reboiler heat duty (QB) MW'])

    fig, (ax10, ax11) = plt.subplots(1, 2)
    fig.suptitle('Feed Stage Sensitivity Analysis')
    fig.set_size_inches(10,4)

    ax10.plot(feed_stage_values, xD_values, 'bo', label = 'xD_benzene')
    ax10.plot(feed_stage_values, xB_tol_values, 'r', label = 'xB_toluene')
    ax10.set_title('Product purity')
    ax10.set_xlabel('feed stage')
    ax10.set_ylabel('mole fraction')

    ax11.plot(feed_stage_values, abs(np.array(Qc_values)), 'go', label = '|QC|')
    ax11.plot(feed_stage_values, Qb_values, 'm--', label = 'QB')
    ax11.plot(feed_stage_values, np.array(Qb_values)+abs(np.array(Qc_values)), 'k', label = 'QB + |QC|')
    ax11.set_title('Heat Duty')
    ax11.set_xlabel('feed stage')
    ax11.set_ylabel('energy (MW)')

    fig.legend(loc = 'right')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(SAVE_DIR, f'feed_stage_sensitivity_{timestamp}.png'))
    plt.show()
    return
  
  def feed_condition_sensitivity_analysis(self,column,q_values=[0, 0.5, 1]):
    """Observe performance sensitivity to feed quality, q. q_values can take custom set of test feed qualities."""

    xD_values = []
    xB_tol_values = []
    Qc_values = []
    Qb_values = []

    for q in q_values:
      outputs = column.run_distillation(q=q,display_output='no')[0]
      xD_values.append(outputs['Benzene purity in distillate (xD)'])
      xB_tol_values.append(outputs['Toluene purity in bottoms (1-xB)'])
      Qc_values.append(outputs['Condenser heat duty (QC) MW'])
      Qb_values.append(outputs['Reboiler heat duty (QB) MW'])

    fig, (ax12, ax13) = plt.subplots(1, 2)
    fig.suptitle('Feed Condition Sensitivity Analysis')
    fig.set_size_inches(10,4)

    ax12.plot(q_values, xD_values, 'bo', label = 'xD_benzene')
    ax12.plot(q_values, xB_tol_values, 'r', label = 'xB_toluene')
    ax12.set_title('Product purity')
    ax12.set_xlabel('feed quality')
    ax12.set_ylabel('mole fraction')
    ax12.set_xticks(q_values)

    ax13.plot(q_values, abs(np.array(Qc_values)), 'go', label = '|QC|')
    ax13.plot(q_values, Qb_values, 'm--', label = 'QB')
    ax13.plot(q_values, np.array(Qb_values)+abs(np.array(Qc_values)), 'k', label = 'QB + |QC|')
    ax13.set_title('Heat Duty')
    ax13.set_xlabel('feed quality')
    ax13.set_ylabel('energy (MW)')
    ax13.set_xticks(q_values)

    fig.legend(loc = 'right')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(SAVE_DIR, f'q_sensitivity_{timestamp}.png'))
    plt.show()
    return

  def process_optimization(self,column,feed_stage_range=(5,12),xD_target=0.95,xB_tol_target=0.95,R_bounds=(1.5, 4.0),q_bounds=(0.0, 1.0),R0=2.5,q0=1):
    """Optmize R and q for each stage and select the set of parameters that result in optimum performance, by minimizing energy consumption whil staying within product purity targets."""

    # Minimize energy consumption
    def objective(variables):
      R, q = variables
      outputs = column.run_distillation(R=R,q=q,feed_stage=feed_stage,display_output='no')[0]
      heat_duty = abs(outputs['Condenser heat duty (QC) MW']) + outputs['Reboiler heat duty (QB) MW']
      return heat_duty

    # Ensure purity targets are met
    def constraint_func(variables):
      R, q = variables
      outputs = column.run_distillation(R=R,q=q,feed_stage=feed_stage,display_output='no')[0]
      xD = outputs['Benzene purity in distillate (xD)']
      xB_tol = outputs['Toluene purity in bottoms (1-xB)']
      return [xD - xD_target, xB_tol - xB_tol_target]


    constraints = [{'type':'ineq', 'fun': lambda v: constraint_func(v)[0]},
     {'type': 'ineq', 'fun': lambda v: constraint_func(v)[1]}]

    bounds = [(R_bounds[0], R_bounds[-1]), # R
    (q_bounds[0], q_bounds[-1])] # q

    stage = []
    min_heat_duty = []
    R_opt = []
    q_opt = []

    # optimization can take a while. Polite messages for reassurance
    wait_messages=["Please wait...","Process is being optimized...","Exploring all possibilities...","Ensuring optimum performance...","Thank you for your patience..."]

    for feed_stage in range(feed_stage_range[0],feed_stage_range[-1]):

      print(wait_messages[feed_stage%len(wait_messages)])    

      result = minimize(
      objective,
      x0=[R0, q0],
      bounds=bounds,
      constraints=constraints,
      method='SLSQP')

      stage.append(feed_stage)
      min_heat_duty.append(result.fun)
      R_opt.append(result.x[0])
      q_opt.append(result.x[1])
    
    min_heat_duty = np.array(min_heat_duty)
    [opt_index] = np.where(min_heat_duty == min_heat_duty.min())
    optimal_setup=[]
    print('\n--OPTIMAL PROCESS PARAMETERS--')
    for i in opt_index:
      print('\nFeed stage = ', stage[i], ', R = ', R_opt[i], ', q = ', q_opt[i])
      optimal_setup.append({'Feed stage': stage[i], 'R': R_opt[i], 'q': q_opt[i]})
    return optimal_setup

class UpsetConditions:

  def process_disturbances(self,column,case_number,feed_comp_values=None,feed_rate_values=None):
    """Analyze column performance and profiles as per disturbances in feed. Use case number 1-3 for predefined cases and case 4 to test custom values of feed composition and flow rate."""

    if case_number == 1:
      feed_comp_values = [0.45, 0.5, 0.55]
      feed_rate_values = [100]
    elif case_number == 2:
      feed_comp_values = [0.5]
      feed_rate_values = [80, 100, 120]
    elif case_number == 3:
      feed_comp_values = [0.45, 0.5, 0.55]
      feed_rate_values = [80, 100, 120]
    else: # default to baseline
      if feed_comp_values is None: feed_comp_values = [0.5]
      if feed_rate_values is None: feed_rate_values = [100]

    fig, (ax14, ax15, ax16) = plt.subplots(1, 3)
    fig.suptitle('Column Profiles')
    ax14.set_title('Temperature')
    ax15.set_title('y_Benzene')
    ax16.set_title('x_Toluene')
    ax14.set_xlabel('Tray #')
    ax15.set_xlabel('Tray #')
    ax16.set_xlabel('Tray #')
    ax14.set_ylabel('temperature (C)')
    ax15.set_ylabel('mole fraction')
    ax16.set_ylabel('mole fraction')
    fig.set_size_inches(12,4)

    Qc=[]
    Qb=[]
    xD=[]
    xB_tol=[]

    for feed_comp in feed_comp_values:
      for feed_rate in feed_rate_values:
        [outputs, Tn, y_real, xn] = column.run_distillation(zF_benzene=feed_comp,F=feed_rate,display_output='no')
        
        ax14.plot(Tn, label = 'zF='+str(feed_comp)+', F='+str(feed_rate))
        ax15.plot(y_real)
        ax16.plot(1-xn)

        Qc.append(outputs['Condenser heat duty (QC) MW'])
        Qb.append(outputs['Reboiler heat duty (QB) MW'])
        xD.append(outputs['Benzene purity in distillate (xD)'])
        xB_tol.append(outputs['Toluene purity in bottoms (1-xB)'])

    fig2, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(abs(np.array(Qc)), 'b', label = '|QC|')
    ax1.plot(Qb, 'r', label = 'QB')
    ax1.plot(abs(np.array(Qc))+np.array(Qb), 'k', label = 'QB+|QC|')
    ax1.set_title('Heat Duty & operational issues')
    ax1.set_xlabel('test number')
    ax1.set_ylabel('energy (MW)')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.plot(xD, 'c', label = 'xD_benzene')
    ax2.plot(xB_tol, 'm', label = 'xB_toluene')
    fig2.legend()
    ax2.set_title('Performance & operational issues')
    ax2.set_xlabel('test number')
    ax2.set_ylabel('mole fraction')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig2.set_size_inches(10,4)
        
    fig.legend(loc = 'right')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(SAVE_DIR, f'column_profiles_disturbances_{timestamp}.png'))
    fig2.savefig(os.path.join(SAVE_DIR, f'performance_disturances{timestamp}.png'))
    plt.show()
    return

  def find_operating_envelope(self,column,xD_target=0.95,xB_tol_target=0.95):
    """Determine allowable error percentages in feed composition and flowrate to ensure product purity targets are met."""

    tol=0.01
    zF_benzene=column.zF_benzene
    i=0

    # find the biggest fluctuation that does not compormise purity.
    while i>=0:
      i+=1

      zF_benzene += tol*i
      [outputs, T_tray, y_real, x_benzene] = column.run_distillation(zF_benzene=zF_benzene,display_output='no')
      if outputs['Benzene purity in distillate (xD)'] < 0.95 or outputs['Toluene purity in bottoms (1-xB)'] < 0.95:
        break
      zF_benzene -= tol*i*2
      [outputs, T_tray, y_real, x_benzene] = column.run_distillation(zF_benzene=zF_benzene,display_output='no')
      if outputs['Benzene purity in distillate (xD)'] < xD_target or outputs['Toluene purity in bottoms (1-xB)'] < xB_tol_target:
        break
    
      zF_benzene=column.zF_benzene

    delta_zF_benzene = abs(zF_benzene-column.zF_benzene)
    zF_benzene=column.zF_benzene
    F=column.F

    j=0
    while j>=0:
      j+=1
      F += j*tol*100
      [outputs, T_tray, y_real, x_benzene] = column.run_distillation(F=F,display_output='no')
      if outputs['Benzene purity in distillate (xD)'] < xD_target or outputs['Toluene purity in bottoms (1-xB)'] < xB_tol_target:
        break
      
      F -= tol*100*j*2
      [outputs, T_tray, y_real, x_benzene] = column.run_distillation(F=F,display_output='no')
      if outputs['Benzene purity in distillate (xD)'] < xD_target or outputs['Toluene purity in bottoms (1-xB)'] < xB_tol_target:
        break
      
      F=column.F

    delta_F = abs(F-column.F)
    F=column.F
    zF_benzene_allowed_error_percent = 100*delta_zF_benzene/zF_benzene
    F_allowed_error_percent = 100*delta_F/F

    print('OPERATING ENVELOPE \nFeed composition: +-' +str(zF_benzene_allowed_error_percent)+'% \nFeed flow rate: +-' +str(F_allowed_error_percent)+'%')
    return {'zF_benzene_allowed_error_percent':zF_benzene_allowed_error_percent, 'F_allowed_error_percent':F_allowed_error_percent}