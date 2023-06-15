import numpy as np
import pandas as pd


class AbstractModelData:
    endog_shape = None
    exog_shape = None
    nobs = None

    # this is the max # of rows that can be requested through get_endog_exog
    max_rows = None

    def xnames(self):
        """
        Column names for exog. If not available, uses ['x1', ...] by default.
        """
        raise NotImplementedError

    def ynames(self):
        """
        Column names for endog. If not available, uses ['y1', ...] by default.
        """
        raise NotImplementedError

    def get_endog_exog(self, start, stop):
        """
        Returns the rows in (endog, exog) indexed by start:stop, as ndarray.
        Ideally this function should not allocate memory, and to allow that it
        can invalidate previous return values.
        """
        raise NotImplementedError


class NdarrayModelData(AbstractModelData):
    def __init__(self, endog, exog, max_rows=None):
        self.endog = np.atleast_2d(endog)
        self.exog = np.atleast_2d(exog)
        self.endog_shape = self.endog.shape
        self.exog_shape = self.exog.shape
        self.nobs = len(self.endog)
        if max_rows is None:
            max_rows = len(self.endog)
        self.max_rows = max_rows

        assert self.endog.ndim == self.exog.ndim == 2
        assert len(self.endog) == len(self.exog)
        assert self.endog.size and self.exog.size

    @classmethod
    def get_names(cls, df, prefix):
        return [prefix + str(i) for i in range(1, df.shape[1] + 1)]

    def xnames(self):
        return self.get_names(self.exog, 'x')

    def ynames(self):
        return self.get_names(self.endog, 'y')

    def get_endog_exog(self, start, stop):
        return self.endog[start:stop, :], self.exog[start:stop, :]


class PandasModelData(AbstractModelData):
    def __init__(self, endog, exog, max_rows=None):
        self.endog = endog
        self.exog = exog
        self.endog_shape = self.endog.shape
        self.exog_shape = self.exog.shape
        self.nobs = len(self.endog)
        if max_rows is None:
            max_rows = len(self.endog)
        self.max_rows = max_rows

        self.endog_scratch = np.empty((max_rows, self.endog_shape[1]))
        self.exog_scratch = np.empty((max_rows, self.exog_shape[1]))

        assert len(self.endog) == len(self.exog)
        assert self.endog.size and self.exog.size

    @classmethod
    def get_names(cls, df, default):
        if isinstance(df, pd.DataFrame):
            if isinstance(df.columns, pd.MultiIndex):
                # flatten MultiIndex
                return ['_'.join((level for level in c if level))
                        for c in df.columns]
            return list(df.columns)
        elif isinstance(df, pd.Series):
            if arr.name:
                return [arr.name]
        return default

    def xnames(self):
        return self.get_names(self.exog, ['x1'])

    def ynames(self):
        return self.get_names(self.endog, ['y1'])

    def get_endog_exog(self, start, stop):
        self.endog_scratch[:stop - start, :] = self.endog.iloc[start:stop]
        self.exog_scratch[:stop - start, :] = self.exog.iloc[start:stop]
        return (self.endog_scratch[:stop - start, :],
                self.exog_scratch[:stop - start, :])

