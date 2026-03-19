
from abc import abstractmethod
import numpy as np

class FuncBase:
    @abstractmethod
    def eval(self, x):
        raise NotImplementedError()

    @property
    @abstractmethod
    def label(self):
        raise NotImplementedError()

    def plot(self, ax, start, stop, num_points=100, logx=False):
        if logx:
            x = np.logspace(np.log10(start), np.log10(stop), num_points)
        else:
            x = np.linspace(start, stop, num_points)

        y = self.eval(x)
        ax.plot(x, y, label=self.label)

class TrackAngularResolution(FuncBase):
    def __init__(self, A, B):
        self._A = A
        self._B = B

    def eval(self, x):
        return self._A + (self._B / x)

    @property
    def label(self):
        return '$%0.2g + \\frac{%0.2g}{p_T}$' % (self._A, self._B)

class TrackPtResolution(FuncBase):
    def __init__(self, A, B):
        self._A = A
        self._B = B

    def eval(self, x):
        return np.sqrt(np.square(self._A) + np.square(self._B * x))

    @property
    def label(self):
        return '$%0.2g \\oplus %0.2g \\cdot p_T$' % (self._A, self._B)