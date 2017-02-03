import sys
from PyQt4 import QtGui

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import skvideo.io

def fig2data ( fig ):
   """
   @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
   @param fig a matplotlib figure
   @return a numpy 3D array of RGBA values
   """
   # draw the renderer
   fig.canvas.draw ( )

   # Get the RGBA buffer from the figure
   w,h = fig.canvas.get_width_height()
   buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
   buf.shape = ( w, h,4 )

   # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
   buf = np.roll ( buf, 3, axis = 2 )
   return buf

class Monitor(FigureCanvas):
    def __init__(self, yf, t_f, videoloc=""):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.i = 0

        self.x = yf[self.i,:(n),0]
        self.y = yf[self.i,:(n),1]

        self.line = self.ax.scatter(self.x,self.y)

        self.fig.canvas.draw()

        self.timer = self.startTimer(.1)
        self.videoloc = videoloc
        self.t_f = t_f
        self.vidwriter = None
        if self.videoloc:
          w,h = self.fig.canvas.get_width_height()
          self.vidwriter = skvideo.io.FFmpegWriter(self.videoloc)

    def timerEvent(self, evt):
        if self.i == self.t_f-1:
          self.i = 0
          self.vidwriter.close()
        # update the height of the bars, one liner is easier
        self.i += 1
        self.x = yf[self.i,:n,0]
        self.y = yf[self.i,:n,1]
        self.ax.cla()

        self.line = self.ax.scatter(self.x,self.y)
        maxx = np.max(self.x) + 20
        minx = np.min(self.x) - 20
        maxy = np.max(self.y) + 20
        miny = np.min(self.y) - 20
        self.ax.set_ylim(ymax=maxy,ymin=miny)
        self.ax.set_xlim(xmin=minx, xmax=maxx)
        self.fig.canvas.draw()
        if self.vidwriter:
          framedata = fig2data(self.fig)
          print framedata.shape
          self.vidwriter.writeFrame(framedata)

from electricity import *
if __name__=="__main__":
  app = QtGui.QApplication(sys.argv)
  w = Monitor(yf, t_f, "./video.mp4")
  w.setWindowTitle("{} Electrical Particle Dynamics".format(r.shape[0]))
  w.show()
  sys.exit(app.exec_())
