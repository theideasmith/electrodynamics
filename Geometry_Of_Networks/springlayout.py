
"""
By embedding a graph in 3+1 dimensions 
we can find a continuous surface on which the 
network lives

This is enabled by a theorem in network science
that the probability of edge collisions
for a graph embedded in three dimensions is 
zero
"""

def spring_layout(y,w):              
    """
    y: an (n*2,d) dimensional matrix where y[:n]_i 
       is the position of the ith node in d dimensions
       and y[n:]_i is the velocity of the ith node
    w: (n,n) matrix of edge weights
    """
    n2 = y.shape[0]
    assert n2%2 ==0

    n = n2/2
    d = y.shape[1]

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
    # Physical distances between nodes                                 
    R = ma.array(                                  
          np.sqrt(np.sum(np.power(dr, 2.), axis=2)),
          mask=nd_identity
        )                               

    # Computing forces using the spring equation
    # this force equation is designed 
    # so that there is no potential
    # energy when R = w
    F = k*(w-R)
    Fnet = np.sum(F, axis=1)
    a = Fnet #nodes have unit mass

    # Setting velocities                               
    y[:n] = np.copy(y[n:])                             
    # Entering the acceleration into the velocity slot 
    y[n:] = np.copy(a)                                 
    # Flattening it out for scipy.odeint to work       
    return np.array(y).reshape(n*2*d)                  
