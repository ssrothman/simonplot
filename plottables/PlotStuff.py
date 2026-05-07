import matplotlib.patches

class AbstractPlotSpec:
    def plot(self, ax):
        raise NotImplementedError()

class LineSpec(AbstractPlotSpec):
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.kwargs = kwargs

    def plot(self, ax):
        ax.plot(self.x, self.y, **self.kwargs)
    
class PointSpec(AbstractPlotSpec):
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.kwargs = kwargs

    def plot(self, ax):
        ax.scatter(self.x, self.y, **self.kwargs)

class BoxSpec(AbstractPlotSpec):
    def __init__(self, x0, y0, width, height, **kwargs):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.kwargs = kwargs

    def plot(self, ax):
        rect = matplotlib.patches.Rectangle((self.x0, self.y0), self.width, self.height, **self.kwargs)
        ax.add_patch(rect)