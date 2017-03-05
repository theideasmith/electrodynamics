import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, yf, numpoints=10):
        self.numpoints = numpoints
        self.stream = self.data_stream()
        self.yf = yf
        self.i=0
        self.n = yf.shape[1]/2
        self.d = yf.shape[2]
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                           init_func=self.setup_plot, blit=True)
                                                   # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

        self.ani.save('video.mp4', writer=writer)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x = self.yf[self.i, :self.n, 0]
        y = self.yf[self.i, :self.n, 1]

        self.scat = self.ax.scatter(x, y,animated=True)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        while True:
            self.i+=1
            if self.i == self.numpoints:
                self.i=0
            yield self.yf[self.i, :self.n, :self.d]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data)

        indmin = max(self.i-10, 0)
        indmax = min(self.i+10, self.yf.shape[0])
        # Set sizes...
        xs = self.yf[indmin:indmax, :self.n, 0]
        ys = self.yf[indmin:indmax, :self.n, 1]
        maxx = np.max(xs) + 20
        minx = np.min(xs) - 20
        maxy = np.max(ys) + 20
        miny = np.min(ys) - 20
        self.ax.set_ylim(ymax=maxy,ymin=miny)
        self.ax.set_xlim(xmin=minx, xmax=maxx)
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def show(self):
        plt.show()
yf = np.load("../data/simulation.npy")
n = yf.shape[1]/2
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(n):
  xs = yf[:,i,0]
  ys = yf[:,i,1]

  ax.plot(xs[1], ys[1], 'gv')
  ax.plot(xs[-1], ys[-1], 'rv')
  ax.plot(xs, ys)

  
xs = yf[:, :n, 0] 
ys = yf[:, :n, 1] 
maxx = np.max(xs) + 20                  
minx = np.min(xs) - 20                  
maxy = np.max(ys) + 20                  
miny = np.min(ys) - 20                  
ax.set_ylim(ymax=maxy,ymin=miny)   
ax.set_xlim(xmin=minx, xmax=maxx)  
plt.savefig("figure-1.jpg")
plt.show()
#tf = yf.shape[0]
#a = AnimatedScatter(yf,numpoints=tf)
#a.show()
