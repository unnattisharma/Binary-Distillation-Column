"""
Balance equations and key relations that describe a binari distillation column under constant molar overflow assumptions.
"""
import numpy as np
from thermodynamics import ThermoModel
import solver
import matplotlib.pyplot as plt
import os
from datetime import datetime

# prepare to save generated figures
SAVE_DIR = 'generated_figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class Column:

  def __init__(self):
    
    # Column
    self.P = 760 # mmHg, atmospheric pressure
    self.N_stages = 15
    self.feed_stage = 8
    self.R = 2.5 # reflux ratio
    self.tray_efficiency_M = 0.7*np.ones(self.N_stages) # tray efficiency for each stage
    self.D = 50 # set default value based on material balance when xD and xB are set equal to target purities

    # Feed
    self.zF_benzene = 0.5 # benzene mole fraction in feed
    self.F = 100 # kmol/hr
    self.q = 1 # quality, i.e., liquid mole fraction in feed

    self.thermo = ThermoModel() #initialze the thermodynamic model

  def calculate_flows(self, F=None, D=None, R=None, q=None):
    """Calculate the flow rates for Bottoms (B), liquid and vapor flow rates for the rectification section (Lr, Vr), and the liquid and vapor flow rates for the stripping section (Ls, Vs) based on the refulx ratio (R), feed quality (q), and the flowrates of feed (F) and distillate (D)."""
    
    if F is None: F = self.F
    if D is None: D = self.D
    if R is None: R = self.R
    if q is None: q = self.q

    B = F - D # bottoms flow rate
    Lr = D*R # constant liquid flow rate in rectifying section
    Vr = D*(1+R) # constant vapor flow rate in stripping section
    Ls  = Lr + q*F # constant liquid flow in stripping section # liquid from F adds to Lr
    Vs = Vr - (1-q)*F # constant vapor flow in stripping section

    flow = {'B': B, 'Lr': Lr, 'Vr': Vr, 'Ls': Ls, 'Vs': Vs}
    return flow

  def y_eq(self,Tn,xn,component,P=None):
    """Calculate the vapor phase mole fraction from the temperature (deg C) and liquid phase mole fraction based on the vapor-liquid equilibrium."""
    
    if P is None: P = self.P
    
    if len(Tn) != len(xn):
      raise Exception("Temperature and composition vector must be the same length")

    if component not in ['benzene', 'toluene']:
      raise ValueError("Invalid component. Enter 'benzene' or 'toluene'.")

    P_sat_n = [self.thermo.calculate_vapor_pressure(T, component) for T in Tn]
    yn = [xn[i]*P_sat_n[i]/P for i in range(len(xn))]
    return yn

  def y_real(self,y_eq_vector,tray_efficiency_M=None,ignore_error_message='no'):
    """Calculate the actual vapor phase mole fraction observed from the predicted equilibrium vapor phase mole fraction based on tray efficiency."""

    if tray_efficiency_M is None: tray_efficiency_M = self.tray_efficiency_M

    
    if ignore_error_message=='no':
      for y in y_eq_vector:
        if y<0 or y>1:
          raise ValueError("Composition out of range.")

    yn = np.ones(len(y_eq_vector))
    yn[-1] = y_eq_vector[-1] # reboiler vapor

    # calculate y_real from bottom to top
    for i in range(len(y_eq_vector)-1, 0, -1):
      yn[i-1] = yn[i] + tray_efficiency_M[i-1]*(y_eq_vector[i-1] - yn[i])
    return yn
  
  def MESH_implementation(self,Tn_xn_vector,component='benzene',N_stages=None,P=None,tray_efficiency_M=None,F=None,D=None,R=None,q=None,feed_stage=None,zF_benzene=None,ignore_error_message='no'):
    """Generate list of residuals (=0 for a solved system) for component material balances and vapor-liquid equilibrium for each stage, including reboiler."""

    if N_stages is None: N_stages = self.N_stages
    if P is None: P = self.P
    if tray_efficiency_M is None: tray_efficiency_M = self.tray_efficiency_M
    if F is None: F = self.F
    if D is None: D = self.D
    if R is None: R = self.R
    if q is None: q = self.q
    if feed_stage is None: feed_stage = self.feed_stage
    if zF_benzene is None: zF_benzene = self.zF_benzene

    #split Tn and xn from the variables vector
    Tn = Tn_xn_vector[:N_stages+1]
    xn = Tn_xn_vector[N_stages+1:]

    if len(Tn) != len(xn):
      print(Tn, xn)
      raise Exception("Temperature and composition vector must be the same length")

    if ignore_error_message=='no':
      for x in xn:
        if x<0 or x>1:
          raise ValueError("Composition out of range.")

    y_eq = self.y_eq(Tn,xn,component,P)
    y_real = self.y_real(y_eq,tray_efficiency_M,ignore_error_message=ignore_error_message)
    [B,Lr,Vr,Ls,Vs] = self.calculate_flows(F, D, R, q).values()

    def material_balances():
      """Material balance for benezene at each stage."""

      mat_bal = []
      mat_bal.append(Vr*y_real[1] + Lr*y_real[0] - Vr*y_real[0] - Lr*xn[0])

      for n in range(1,len(xn)-1):
        if n==feed_stage-1:
          mat_bal.append(Vs*y_real[n+1] + Lr*xn[n-1] + F*zF_benzene - Vr*y_real[n] - Ls*xn[n])
        else:
          if n<feed_stage-1:
            mat_bal.append(Vr*y_real[n+1] + Lr*xn[n-1] - Vr*y_real[n] - Lr*xn[n])
          else:
            mat_bal.append(Vs*y_real[n+1] + Ls*xn[n-1] - Vs*y_real[n] - Ls*xn[n])

      mat_bal.append(Ls*xn[n] - Vs*y_real[n+1] - B*xn[n+1])
      return mat_bal

    def equilibrium_equations():
      """Apply vapor-liquid equilibrium for each stage."""

      equil_eq = []
      equil_eq.append(y_eq[0] + self.y_eq([Tn[0]], [1-xn[0]], 'toluene')[0] - 1)

      for n in range(1,len(xn)-1):
        equil_eq.append(y_eq[n] + self.y_eq([Tn[n]], [1-xn[n]], 'toluene')[0] - 1)

      equil_eq.append(y_eq[n+1] + self.y_eq([Tn[n+1]], [1-xn[n+1]], 'toluene')[0] - 1)
      return equil_eq

    MESH_equations = material_balances()+equilibrium_equations()
    return MESH_equations

  def column_outputs(self,Tn_soln,xn_soln,component='benzene',N_stages=None,P=None,tray_efficiency_M=None,F=None,D=None,R=None,q=None,feed_stage=None,zF_benzene=None,display_output='no'):
    """Determine heat duties and purities from temperature (deg C) and composition profiles (liquid mole fraction of benzene), and visualize column profiles."""

    if N_stages is None: N_stages = self.N_stages
    if P is None: P = self.P
    if tray_efficiency_M is None: tray_efficiency_M = self.tray_efficiency_M
    if F is None: F = self.F
    if D is None: D = self.D
    if R is None: R = self.R
    if q is None: q = self.q
    if feed_stage is None: feed_stage = self.feed_stage
    if zF_benzene is None: zF_benzene = self.zF_benzene

    if len(Tn_soln) != len(xn_soln):
      raise Exception("Temperature and composition vector must be the same length")

    for x in xn_soln:
      if x<0 or x>1:
        raise ValueError("Composition out of range.")

    y_eq = self.y_eq(Tn_soln,xn_soln,component,P)
    y_real = self.y_real(y_eq,tray_efficiency_M)
    [B,Lr,Vr,Ls,Vs] = self.calculate_flows(F, D, R, q).values()
    xD = y_real[0]
    xB = xn_soln[-1]

    TD = self.thermo.calculate_bubble_point(xD, 1-xD,P=P) # distillate temperature is the bubble point
    QB = (B*self.thermo.Cp_mix('L',xB)*Tn_soln[-1] + Vs*(self.thermo.Cp_mix('V',y_real[-1])*Tn_soln[-1] +  self.thermo.h_mix(y_real[-1])) - Ls*self.thermo.Cp_mix('L', xn_soln[N_stages-1])*Tn_soln[N_stages-1])/3600000
    QC = ((Lr + D)*self.thermo.Cp_mix('L',xD)*TD - Vr*(self.thermo.Cp_mix('V',y_real[0])*Tn_soln[0] + self.thermo.h_mix(y_real[0])))/3600000

    outputs = {'Benzene purity in distillate (xD)': y_real[0],
                'Toluene purity in bottoms (1-xB)': 1-xn_soln[-1],
                'Condenser heat duty (QC) MW': QC,
                'Reboiler heat duty (QB) MW': QB}

    if display_output == 'yes':
      fig, (ax3, ax4, ax5) = plt.subplots(1, 3)
      fig.suptitle('Column Profiles')
      ax3.plot(Tn_soln, 'k', label = 'Tray temperature')
      ax4.plot(y_real, 'bo', label = 'Benzene mole fraction in vapor')
      ax5.plot(1-xn_soln, 'r', label = 'Toluene mole fraction in liquid')
      ax3.set_title('Temperature')
      ax4.set_title('y_Benzene')
      ax5.set_title('x_Toluene')
      ax3.set_xlabel('Tray #')
      ax4.set_xlabel('Tray #')
      ax5.set_xlabel('Tray #')
      ax3.set_ylabel('temperature (C)')
      ax4.set_ylabel('mole fraction')
      ax5.set_ylabel('mole fraction')
      fig.set_size_inches(12,4)
      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
      fig.savefig(os.path.join(SAVE_DIR, f'column_profiles_{timestamp}.png'))
      plt.show()
      [print(output+': '+str(outputs[output])+'\n') for output in outputs]
    return [outputs, Tn_soln, y_real, xn_soln]

  def run_distillation(self,N_stages=None,P=None,tray_efficiency_M=None,F=None,D=None,R=None,q=None,feed_stage=None,zF_benzene=None,display_output='no',initial_guess=None):
    """Determine column profiles (temperature and composition) for set column conditions."""
    
    if N_stages is None: N_stages = self.N_stages
    if P is None: P = self.P
    if tray_efficiency_M is None: tray_efficiency_M = self.tray_efficiency_M
    if F is None: F = self.F
    if D is None: D = self.D
    if R is None: R = self.R
    if q is None: q = self.q
    if feed_stage is None: feed_stage = self.feed_stage
    if zF_benzene is None: zF_benzene = self.zF_benzene
    
    if initial_guess is None:
      initial_guess = np.array([100]*(N_stages+1)+[0.5]*(N_stages+1)) # guess all tray temp = 100, all x_benzene = 0.5
    
    column_solutions = solver.newton_raphson(initial_guess, self.MESH_implementation,N_stages=N_stages,P=P,tray_efficiency_M=tray_efficiency_M,F=F,D=D,R=R,q=q,feed_stage=feed_stage,zF_benzene=zF_benzene)
    Tn = column_solutions[0][0:N_stages+1]
    xn = column_solutions[0][N_stages+1::]
    results = self.column_outputs(Tn,xn,N_stages=N_stages,P=P,tray_efficiency_M=tray_efficiency_M,F=F,D=D,R=R,q=q,feed_stage=feed_stage,zF_benzene=zF_benzene,display_output=display_output)
    return results