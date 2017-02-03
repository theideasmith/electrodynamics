import numpy as np
from scipy.integrate import odeint
mag = lambda r: np.sqrt(np.sum(np.power(r, 2)))
k = 8.99*10**9


def g(r, t, q1, q2, m1, m2):
 r1, dr1dt, r2, dr2dt, r3, dr3dt = r.reshape((6,2))
 F12 = q1*q2/mag(r2-r1)**2
 F23 = q1*q2/mag(r2-r3)**2
 F31 = q1*q2/mag(r3-r1)**2

 dy = [
     dr1dt,
     (F12/m1)*(r1-r2)+(F31/m1)*(r1-r3),
     dr2dt,
     (F12/m2)*(r2-r1)+(F23/m1)*(r2-r3),
     dr3dt,
     (F31/m2)*(r3-r1)+(F23/m1)*(r3-r2)
 ]
 return np.array(dy).reshape(12)

t = np.linspace(0, 100, num=10000)
r1i = np.array([1.,2.])
dr1dti = np.array([2,.4])

r2i = np.array([3.,4.])
dr2dti = np.array([0.3, -2.])

r3i = np.array([-2.,1.])
dr3dti = np.array([-0.3, -2.])

y0 = np.array([r1i, dr1dti, r2i, dr2dti, r3i, dr3dti]).reshape(12)

# We are keeping the constants to be 1 because I am lazy
# the qualitative behavior doesn't change
y = odeint(g, y0, t, args=(-1,1,1,1)).reshape(10000,6,2)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
ys1 = y[:,0,1]
xs1 = y[:,0,0]

xs2 = y[:,2,0]
ys2 = y[:,2,1]

xs3 = y[:,4,0]
ys3 = y[:,4,1]

"""
ax.plot(xs1[:1], ys1[:1], t[:1],'bv')
ax.plot(xs1[-1:], ys1[-1:], t[-1:], 'rv')
ax.plot(xs2[:1], ys2[:1], t[:1], 'bv')
ax.plot(xs2[-1:], ys2[-1:], t[-1:], 'rv')

ax.plot(xs3[:1], ys3[:1], t[:1], 'bv')
ax.plot(xs3[-1:], ys3[-1:], t[-1:], 'rv')

ax.plot(xs1, ys1, t)
ax.plot(xs2, ys2, t)
ax.plot(xs3, ys3, t)
"""


ax.plot(xs1[:1], ys1[:1],'bv')     
ax.plot(xs1[-1:], ys1[-1:], 'rv') 
ax.plot(xs2[:1], ys2[:1], 'bv')    
ax.plot(xs2[-1:], ys2[-1:], 'rv') 
                                          
ax.plot(xs3[:1], ys3[:1], 'bv')    
ax.plot(xs3[-1:], ys3[-1:], 'rv') 
                                          
ax.plot(xs1, ys1)                      
ax.plot(xs2, ys2)                      
ax.plot(xs3, ys3)                      

plt.show()
