import numpy as np
import numpy.ma as ma
from scipy.integrate import odeint
mag = lambda r: np.sqrt(np.sum(np.power(r, 2)))

def mutual_distance_vectors(y):
    n = y.shape[0]
    # rj across, ri down
    rs_from = np.tile(y, (n,1,1))
    # ri across, rj down
    rs_to = np.transpose(rs_from, axes=(1,0,2))
    # directional distance between each r_i and r_j
    # dr_ij is the force from j onto i, i.e. r_i - r_j
    dr = rs_to - rs_from
    dr = dr.astype(np.float32)
    return dr

def mutual_distances(y):

    dR = mutual_distance_vectors(y)
    R = np.array(
      np.power(
        np.sum(np.power(dR, 2.), axis=2),
        1./2.
      )
    ).astype(np.float32)
    return R

def spring_layout(y,t,w,k,n,d,T):
  """
  y: an (n*2,d) dimensional matrix where y[:n]_i
     is the position of the ith node in d dimensions
     and y[n:]_i is the velocity of the ith node
  w: (n,n) matrix of edge weights
  """
  y = np.copy(y.reshape((n*2,d)))
  x = y[:n]
  v = y[n:]
  dR = mutual_distance_vectors(x)

  # F=0 <=> R=w
  # we also add a damping term
  F = -k*(dR-w*dR/(np.linalg.norm(dR)))
  Fnet = np.sum(F, axis=1) - v
  a = Fnet #nodes have unit mass
  # Setting velocities
  y[:n] = np.copy(y[n:])
  # Entering the acceleration into the velocity slot
  y[n:] = np.copy(a)
  # Flattening it out for scipy.odeint to work
  return np.array(y).reshape(n*2*d)

def integrator_func(y, t, q, m, n, d, k):
    y = np.copy(y.reshape((n*2,d)))
    x = y[:n]
    v = y[n:]
    dR = mutual_distance_vectors(x)

    R = ma.array(
        mutual_distances(x)
        mask = np.eye(n).reshape((n,n,1))
    )

    # Pairwise q_i*q_j for force equation
    qsa = np.tile(q, (n,1))
    qsb = np.tile(q, (n,1)).T
    qs = qsa*qsb

    # Electrical Forces forces
    Fs = k*(qs/drmag).reshape((n,n, 1))*dr

    # Dividing by m to obtain acceleration vectors
    Fnet = np.sum(Fs, 1)
    a = Fnet/m

    # Setting velocities
    y[:n] = np.copy(v)

    # Entering the acceleration into the velocity slot
    y[n:] = np.copy(a)

    # Flattening it out for scipy.odeint to work
    return np.array(y).reshape(n*2*d)

def sim_particles(t, r, v, q, m, k=1.):

    d = r.shape[-1]
    n = r.shape[0]
    y0 = np.zeros((n*2,d))
    y0[:n] = r
    y0[n:] = v
    y0 = y0.reshape(n*2*d)
    yf = odeint(
        integrator_func,
        y0,
        t,
        args=(q,m,n,d,k)).reshape(t.shape[0],n*2,d)
    return yf

if __name__=="__main__":
    t = 20
    t_f = t*10
    t = np.linspace(0, t, num=t_f)
    from situations import chosen
    r,v,q,m = chosen()
    yf = sim_particles(t,r,v,q,m)
    np.save('../data/simulation.npy', yf)
