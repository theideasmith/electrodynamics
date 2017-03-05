import numpy as np
import numpy.ma as ma
from scipy.integrate import odeint
mag = lambda r: np.sqrt(np.sum(np.power(r, 2)))

def integrator_func(y, t, q, m, n, d, k):
    y = np.copy(y.reshape((n*2,d)))
    # rj across, ri down
    rs_from = np.tile(y[:n], (n,1,1))
    # ri across, rj down
    rs_to = np.transpose(rs_from, axes=(1,0,2))
    # directional distance between each r_i and r_j
    # dr_ij is the force from j onto i, i.e. r_i - r_j
    dr = rs_to - rs_from
    # Used as a mask
    nd_identity = np.eye(n).reshape((n,n,1))
    # Force magnitudes
    drmag = ma.array(
        np.power(
            np.sum(np.power(dr, 2.), 2.)
        ,3./2.)
      ,mask=nd_identity)
    print "---------"
    # Pairwise q_i*q_j for force equation
    qsa = np.tile(q, (n,1))
    qsb = np.tile(q, (n,1)).T
    qs = qsa*qsb
    # Directional forces
    Fs = k*(qs/drmag).reshape((n,n, 1))*dr
    # Dividing by m to obtain acceleration vectors
    Fnet = np.sum(Fs, 1)
    a = Fnet
    # Setting velocities
    y[:n] = np.copy(y[n:])
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
