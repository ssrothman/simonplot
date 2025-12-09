from .SetupConfig import config
import hist

def transform_from_string(str):
    if str is None or str.lower() == "none":
        return None
    else:
        return getattr(hist.axis.transform, str)

class AbstractBinning:
    def build_axis(self, variable):
        raise NotImplementedError()

class DefaultBinning(AbstractBinning):
    def __init__(self):
        pass

    def build_axis(self, variable):
        cfg = config['default_binnings'][variable.key]
        if cfg['type'] == 'regular':
            return RegularBinning(
                nbins=cfg['nbins'],
                low=cfg['low'],
                high=cfg['high'],
                transform=cfg.get('transform', None)
            ).build_axis(variable)
        elif cfg['type'] == 'explicit':
            return ExplicitBinning(
                edges=cfg['edges']
            ).build_axis(variable)
        else:
            raise ValueError("Unknown binning type: %s"%(cfg['type']))

class RegularBinning(AbstractBinning):
    def __init__(self, nbins, low, high, transform=None):
        self._nbins = nbins
        self._low = low
        self._high = high
        if type(transform) is str:
            self._transform = transform_from_string(transform)
        else:
            self._transform = transform
    @property
    def nbins(self):
        return self._nbins
    
    @property
    def low(self):
        return self._low
    
    @property
    def high(self):
        return self._high

    @property
    def transform(self):
        return self._transform
    
    def build_axis(self, variable):
        return hist.axis.Regular(
            self.nbins,
            self.low,
            self.high,
            transform=self.transform,
            name=variable.key,
            label=config['axis_labels'].get(variable.key, variable.key)
        )
    
class ExplicitBinning(AbstractBinning):
    def __init__(self, edges):
        self._edges = edges

    @property
    def edges(self):
        return self._edges
    
    def build_axis(self, variable):
        return hist.axis.Variable(
            self.edges,
            name=variable.key,
            label=config['axis_labels'].get(variable.key, variable.key)
        )