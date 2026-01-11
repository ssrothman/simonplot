from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds

import numpy as np
import awkward as ak

import hist
import matplotlib.axes


from simonpy.AbitraryBinning import ArbitraryBinning

from typing import List, Union, override

from .DatasetBase import SingleDatasetBase, DatasetStackBase
from simonplot.typing.Protocols import BaseDatasetProtocol

class DatasetStack(DatasetStackBase):
    def __init__(self, key : str, color : str | None, label : str, datasets : list[BaseDatasetProtocol]):
        self._key = key
        self._color = color
        self._label = label
        self._datasets = datasets
        
class NanoEventsDataset(SingleDatasetBase):
    def __init__(self, key : str, color : str | None, label : str, fname, **options):
        self._key = key
        self._color = color
        self._label = label

        #suppress warnings
        NanoAODSchema.warn_missing_crossrefs = False

        import coffea
        version = coffea._version.version_tuple
        if int(version[0]) >= 2025 and int(version[1]) >= 11 and int(version[2]) >= 0:
            options['mode'] = 'virtual'
        else:
            options['delayed'] = False

        self._events = NanoEventsFactory.from_root(
            fname,
            **options 
        ).events()
        
    def ensure_columns(self, columns):
        # NanoEvents loads all columns on demand, so nothing to do here
        pass

    def get_column(self, column_name, collection_name=None):
        if '.' in column_name:
            raise ValueError("NanoEventsDataset.get_column: column_name '%s' contains '.'! Instead use collection_name argument."%(column_name))
        
        if collection_name is not None:
            return ak.materialize(self._events[collection_name][column_name])
        else:
            return ak.materialize(self._events[column_name])
        
    @property
    def num_rows(self):
        return len(self._events)
    
class ParquetDataset(SingleDatasetBase):
    def __init__(self, key : str, color : str | None, label : str, path, filesystem=None):
        self._key = key
        self._color = color
        self._label = label

        self._dataset = ds.dataset(path, format="parquet", filesystem=filesystem)
            
    def ensure_columns(self, columns):
        has_everything = True
        if hasattr(self, '_table'):
            for col in columns:
                if col not in self._table.column_names:
                    has_everything = False
                    break
        else:
            has_everything = False

        if not has_everything:
            self._table = self._dataset.to_table(columns=columns)
    
    def get_column(self, column_name, collection_name=None):
        if collection_name is not None:
            raise NotImplementedError("ParquetDataset does not support collection_name argument")
        
        if not hasattr(self, '_table'):
            raise RuntimeError("ParquetDataset.ensure_columns must be called before get_column")
        
        if column_name not in self._table.column_names:
            raise RuntimeError("Column %s not loaded! Call ensure_columns() first"%column_name)

        return self._table[column_name].to_numpy()
    
    @property
    def num_rows(self):
        if hasattr(self, '_table'):
            return self._table.num_rows
        else:
            return self._dataset.count_rows()
    
    #extra properties for parquetdatasets for utility
    @property
    def files(self):
        return self._dataset.files
    
    @property 
    def filesystem(self):
        return self._dataset.filesystem
    
    @property
    def schema(self):
        return self._dataset.schema