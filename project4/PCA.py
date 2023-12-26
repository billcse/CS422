import numpy as np
import matplotlib.pyplot as plt

def compute_covariance_matrix(Z):
  return np.cov(Z.T)

def find_pcs(cov):
  L, pcs = np.linalg.eig(cov)
  index = np.flip(np.argsort(L))
  L = L[index]
  pcs[:, index]
  return pcs, L
  
def project_data(Z, pcs, L):
  result = pcs[:, 0]
  Z_star = []
  for i in Z:
    projMatrix = np.dot(i, result)
    Z_star.append(projMatrix)
  return Z_star

def show_plot(Z, Z_star):
  samples = len(Z_star)
  plt.scatter(Z[:, 0], Z[:, 1])
  plt.scatter(Z_star, np.zeros(samples))
  plt.show()

