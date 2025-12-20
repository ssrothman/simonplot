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