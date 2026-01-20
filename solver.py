"""
Implementation of the Newton Raphson method of numerical solution.
"""
import numpy as np

def perturb(v_vector, i, epsilon=1e-5):
  """Perturb the i-th element of v_vector by epsilon."""
  
  v_perturbed = v_vector.copy()
  v_perturbed[i] += epsilon
  return v_perturbed

def calculate_jacobian(variables_vector,residuals_function,N_stages,P,tray_efficiency_M,F,D,R,q,feed_stage,zF_benzene,current_func_val=None,epsilon=1e-5):
  """Calculate the Jacobian matrix for a residual function through perturbation of the variables by epsilon."""

  # ensure f1 is calculated
  if current_func_val is None:
    current_func_val = residuals_function(variables_vector,N_stages=N_stages,P=P,tray_efficiency_M=tray_efficiency_M,F=F,D=D,R=R,q=q,feed_stage=feed_stage,zF_benzene=zF_benzene,ignore_error_message='yes') # calculate f
  
  N = len(variables_vector)
  J = np.zeros((N, N))

  # loop to calculate df_i/dx_j
  for i in range(N):
    for j in range(N):
      perturbed_func_val = residuals_function(perturb(variables_vector, j,epsilon),N_stages=N_stages,P=P,tray_efficiency_M=tray_efficiency_M,F=F,D=D,R=R,q=q,feed_stage=feed_stage,zF_benzene=zF_benzene,ignore_error_message='yes') # perturb variables # calculate f2
      J[i][j] = (np.array(perturbed_func_val[i]) - np.array(current_func_val[i]))/epsilon
  return J

def newton_raphson(variables_vector,residuals_function,N_stages,P,tray_efficiency_M,F,D,R,q,feed_stage,zF_benzene,max_iter=10,tol=1e-3,epsilon=1e-5):
  """Apply the Newton-Raphson method to solve for a variables_vector that sets the residuals_function to zero."""

  iter = 0
  delta = 1
  while iter <= max_iter and delta > tol:
    iter += 1

    f1 = residuals_function(variables_vector,N_stages=N_stages,P=P,tray_efficiency_M=tray_efficiency_M,F=F,D=D,R=R,q=q,feed_stage=feed_stage,zF_benzene=zF_benzene,ignore_error_message='yes') # calculate f1
    J = calculate_jacobian(variables_vector,residuals_function,N_stages=N_stages,P=P,tray_efficiency_M=tray_efficiency_M,F=F,D=D,R=R,q=q,feed_stage=feed_stage,zF_benzene=zF_benzene,current_func_val=f1,epsilon=1e-5)
    
    variables_next = variables_vector - np.linalg.inv(J)@f1 # refine variable vector values

    delta = abs(np.linalg.norm(variables_vector)-np.linalg.norm(variables_next))
    variables_vector = variables_next
  return [variables_vector,residuals_function(variables_vector),iter,delta]