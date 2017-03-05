import numpy as np
def two_dipoles():
  a = 0.
  b = 100.
  r = np.array([
      [a, 0.],
      [a+20.,0.],
  ])
  v = np.array([
      [2.,10.],
      [-2., -10.],


  ])
  q = np.array([
      20.,
     -20.,
  ])
  n = r.shape[0]
  m = np.ones(n)
  return r,v,q,m

def collision():
  r = np.array([
      [0., 0.5],
      [100.,0.],

  ])
  v = np.array([
      [10.,0.],
      [-100., 0.],
  ])
  q = np.array([
      20.,
      20.,
  ])
  n = r.shape[0]
  m = np.ones(n)
  return r,v,q,m

chosen = two_dipoles
