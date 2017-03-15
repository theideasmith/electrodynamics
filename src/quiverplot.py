from mayavi.mlab import *
import numpy as np


def gen_vectors(f,x,y,z):
    assert len(x) == len(y)

    d = len(x), len(y)
    t = len(x)
    res = []
    xs = []
    ys = []
    zs = []
    """
    Complexity grows proportional to N^2
    - which makes sense
    """
    for i in xrange(len(x)):
        for j in xrange(len(y)):
            for k in xrange(len(z)):
                res.append(f(x[i], y[j], z[k]))
                xs.append(x[i])
                ys.append(y[j])
                zs.append(z[k])

    res = np.array(res)
    return np.array(xs), np.array(ys), np.array(zs), res[:,0], res[:,1], res[:,2]

def quivers(f,xlim=None,ylim=None, zlim=None, num=10):
    """
    ======== ABOUT ==========
    Generates an dx*dy*dz matrix
    by applying f on all x[i],y[j],z[k] pairs
    =========================
    """

    if not xlim: xlim =(-1,2)
    if not ylim: ylim=(-1,2)
    if not zlim: zlim=(-1,1)

    assert type(num) ==int or type(num) == tuple
    assert type(xlim)==tuple and type(ylim)==tuple

    if type(num) == int:
        num = (num, num, num)

    xs = np.linspace(*xlim, num=num[0])
    ys = np.linspace(*ylim, num=num[1])
    zs = np.linspace(*zlim, num=num[2])
    print xs == zs
    X,Y,Z, U,V,W = gen_vectors(f, xs, ys, zs)
    return X,Y,Z, U,V,W

def dipole(x,y,z):
    r = np.array([
        [0,0,0],
        [1,1,0]])

    X = np.array([x,y,z])
    X = np.tile(X, (2,1))
    d = (r-X)
    d/=np.linalg.norm(d, axis=1, keepdims=True)
    d[0]*=-1

    R= np.sum(d, axis=0)
    return R/np.linalg.norm(R)

if __name__=="__main__":
    x,y,z, u,v,w = quivers(dipole, num=10)
    m = np.hypot(u,v,w)
    print m.shape
    obj = quiver3d(x, y, z, u, v, w, line_width=1, scale_factor=0.1)
    show()
