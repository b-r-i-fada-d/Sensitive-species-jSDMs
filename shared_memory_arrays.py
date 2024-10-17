
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
from pandas import Index, MultiIndex


class SharedNumpyArray:
    """Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing."""

    def __init__(self, array):
        self._shared = SharedMemory(create=True, size=array.nbytes)
        self._dtype, self._shape = array.dtype, array.shape

        # Initialize a new numpy array that uses the shared memory and copy data from
        # the array to this shared memory.
        res = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared.buf)
        res[:] = array[:]

    def read(self):
        """Read array from shared memory."""
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        """Copy array stored in shared memory."""
        return np.copy(self.read())

    def unlink(self):
        """Release the allocated memory. Call when finished using the data,
        or when the data is copied somewhere else."""
        self._shared.close()
        self._shared.unlink()


class SharedPandasDataFrame:
    """Wraps a Pandas dataframe so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing."""

    def __init__(self, df):
        self._values = SharedNumpyArray(df.values)
        self._index = SharedNumpyArray(df.index.to_numpy())
        self._index_names = list(df.index.names)
        self._is_multiindexed = isinstance(df.index, MultiIndex)
        self._columns = list(df.columns)

    def read(self):
        if self._is_multiindexed:
            index = MultiIndex.from_tuples(self._index.read(), names=self._index_names)
        else:
            index = Index(self._index.read(), name=self._index_names[0])
        return pd.DataFrame(self._values.read(), index=index, columns=self._columns)

    def copy(self):
        index = MultiIndex.from_tuples(self._index.read(), names=self._index_names)
        return pd.DataFrame(self._values.copy(), index=index, columns=self._columns)

    def unlink(self):
        self._values.unlink()
        self._index.unlink()
