""""
This code will computationally study raytracing, that is
projecting space points onto a plane such that they can be
rendered on a graphics processor. This will enable me 
to build my own raytracer. 

I'll be able to implement a 3D graphics plotter
that renders onto a matrix that I then show using
plt.imshow

First definitions:

Let 

v be a viewpoint vector
q be an arbitrary point in space
r = v-q
l = r*t+v where t is a scalar \in R
n be a normal vector to a plane
w be an arbitrary point lying on the plane

An arbitrary plane is given by 
  (l - w).n = 0                    (1) 

Substituting in l=r*t+v:
        (r*t + v-w).n = 0          (2)
    t*(r.n) + (v-w).n = 0
              t*(r.n) = (w-v).n
                       (w-v).n
                  t = ----------
                        (r.n) 

We have thus derived an expression for the parameter t
Now let us compute the point given by t:
     l = r*t + v
"""

def project(v, q, r, l, n, w):
  t = (w-v).dot(n)/r.dot(n)
  return l*t+v


