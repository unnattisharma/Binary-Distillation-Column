"""
Thermodynamic laws governing operation of binary distillation column.
"""
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# prepare to save generated figures
SAVE_DIR = 'generated_figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class ThermoModel:

  def __init__(self):
    self.hv_benzene = 104 #J/mol K
    self.hv_toluene = 130 #J/mol K
    self.hl_benzene = 135.1 #J/mol K
    self.hl_toluene = 167 #J/mol K
    self.hvap_benzene = 34 #kJ/mol
    self.hvap_toluene = 37 #kJ/mol
    self.P = 760 #mm Hg
    self.antoine = {'benzene' : {'A': 6.90565, 'B': 1211.033, 'C': 220.79},
                    'toluene' : {'A': 6.95334, 'B': 1343.943, 'C': 219.377}}

  def calculate_vapor_pressure(self, T, component):
    """Use Antoine equation to calculate vapor pressure (mmHg) of a component, given temperature T (deg C)."""
    
    if component not in ['benzene', 'toluene']:
      raise ValueError("Invalid component. Enter 'benzene' or 'toluene'.")

    return 10**(self.antoine[component]['A'] - self.antoine[component]['B']/(self.antoine[component]['C'] + T))

  def calculate_bubble_point(self, x_benzene, x_toluene, P=None):
    """Use Antoine equation to solve for bubble point (deg C) at pressure P (mmHg), given component mole fractions in liquid phase."""

    if P is None: P = self.P
    if x_benzene<0 or x_benzene>1 or x_toluene<0 or x_toluene>1:
      raise ValueError("Composition out of range.")

    def bubble_point_equation(T):
      P_sat_benzene = self.calculate_vapor_pressure(T, 'benzene')
      P_sat_toluene = self.calculate_vapor_pressure(T, 'toluene')
      P_bubble = x_benzene*P_sat_benzene + x_toluene*P_sat_toluene
      objective = P_bubble - P
      return objective

    T_bubble = fsolve(bubble_point_equation, 100) # initial guess: T=100 deg C
    return T_bubble[0]

  def calculate_dew_point(self, y_benzene, y_toluene, P=None):
    """Use Antoine equation to solve for dew point (deg C) at pressure P (mmHg), given component mole fraction in vapor phase."""

    if P is None: P = self.P
    if y_benzene<0 or y_benzene>1 or y_toluene<0 or y_toluene>1:
      raise ValueError("Composition out of range.")
    def dew_point_equation(T):
      P_sat_benzene = self.calculate_vapor_pressure(T, 'benzene')
      P_sat_toluene = self.calculate_vapor_pressure(T, 'toluene')
      objective = y_benzene*P/P_sat_benzene + y_toluene*P/P_sat_toluene - 1
      return objective

    T_dew = fsolve(dew_point_equation, 100) # initial guess: T=100 deg C
    return T_dew[0]

  def generate_VLE(self, P=None):
    """Use Antoine-based bubble point and vapor pressure to generate VLE diagram."""

    if P is None: P = self.P
    x_benzene = np.linspace(0, 1 , 10)
    x_toluene = 1 - x_benzene
    y_benzene = []
    y_toluene = []
    T_vals = []
    for x in x_benzene:
      T_bubble = self.calculate_bubble_point(x, 1-x, P)
      T_vals.append(T_bubble)
      y_benzene.append(x*self.calculate_vapor_pressure(T_bubble, 'benzene')/P)
      y_toluene.append((1-x)*self.calculate_vapor_pressure(T_bubble, 'toluene')/P)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('VLE diagram')
    ax1.plot(x_benzene, T_vals, 'b', label = 'benzene_liquid')
    ax1.plot(y_benzene, T_vals, 'bo', label = 'benzene_vapor')
    ax2.plot(x_toluene, T_vals, 'r', label = 'toluene_liquid')
    ax2.plot(y_toluene, T_vals, 'ro', label = 'toluene_vapor')
    ax1.set_title('benzene')
    ax2.set_title('toluene')
    ax1.set_xlabel('benzene mole fraction')
    ax2.set_xlabel('toluene mole fraction')
    ax1.set_ylabel('temperature (C)')
    ax2.set_ylabel('temperature (C)')
    fig.set_size_inches(9,3.5)
    fig.legend(loc = 'right')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(SAVE_DIR, f'vle_diagram_{timestamp}.png'))
    plt.show()
    return

  def property_mix(self,h1, h2, x1):
    """calculate property of mixture from composition and component properties."""
    
    if x1<0 or x1>1:
      raise ValueError("Composition out of range.")
    return x1*h1 + (1-x1)*h2

  def Cp_mix(self,phase, comp_ben):  #kJ/kmol.K
    """calculate specific heat capacity (kJ/kmol.K) of mixture based on the phase."""

    if comp_ben<0 or comp_ben>1:
      raise ValueError("Composition out of range.")

    if phase == 'L':
      return self.property_mix(self.hl_benzene, self.hl_toluene, comp_ben)
    if phase == 'V':
      return self.property_mix(self.hv_benzene, self.hv_toluene, comp_ben)

  def h_mix(self,comp_ben): #kJ/kmol
    """find enthalpy of vaporization (kJ/kmol) for mixture."""
    
    if comp_ben<0 or comp_ben>1:
        raise ValueError("Composition out of range.")

    return self.property_mix(self.hvap_benzene, self.hvap_toluene, comp_ben)*1000