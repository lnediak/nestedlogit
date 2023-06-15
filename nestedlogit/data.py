import numpy as np
import pandas as pd


def generic_get_names(df, prefix='', default=None):
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            # flatten MultiIndex
            return ['_'.join((level for level in c if level))
                    for c in df.columns]
        return list(df.columns)
    elif isinstance(df, pd.Series):
        if arr.name:
            return [arr.name]
        return [prefix + '1']
    elif isinstance(df, np.ndarray):
        return [prefix + str(i) for i in range(1, df.shape[1] + 1)]
    try:
        return list(df)
    except TypeError:
        pass
    return default


class AbstractModelData:
    endog_shape = NotImplemented
    exog_shape = NotImplemented
    nobs = NotImplemented

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
        It is assumed that 0 <= start,stop <= self.nobs.

        Ideally this function should not allocate memory, and to allow that it
        can invalidate previous return values.

        Also note that this will only be called (by the model) in a sequence
        resulting in a sweep over the data (e.g. 0:100, 100:200, 200:300, etc),
        possibly going through many times.
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

    def xnames(self):
        return generic_get_names(self.exog, prefix='x')

    def ynames(self):
        return generic_get_names(self.endog, prefix='y')

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

    def xnames(self):
        return generic_get_names(self.exog, prefix='x')

    def ynames(self):
        return generic_get_names(self.endog, prefix='y')

    def get_endog_exog(self, start, stop):
        self.endog_scratch[:stop - start, :] = self.endog.iloc[start:stop]
        self.exog_scratch[:stop - start, :] = self.exog.iloc[start:stop]
        return (self.endog_scratch[:stop - start, :],
                self.exog_scratch[:stop - start, :])


class IndexedModelData(AbstractModelData):
    """
    Here, exog = concatenate(
              [tables[table_name].iloc[exog_inds[table_name], :]
               for table_name in exog_inds], axis=1)
    and endog is similar.
    """

    def __init__(self, endog_inds, exog_inds, tables, max_rows=None):
        """
        Note that tables, as passed in, needs to be a dict-like with values
        that can be passed into pandas.DataFrame.

        endog_inds/exog_inds can be passed in as an ndarray, in which case
        the columns will be ['y1'/'x1', etc.] which will be keys in tables.
        """
        self.endog_inds = pd.DataFrame(
            endog_inds, columns=generic_get_names(endog_inds, prefix='y'))
        self.exog_inds = pd.DataFrame(
            exog_inds, columns=generic_get_names(exog_inds, prefix='x'))
        self.tables = {table_name: pd.DataFrame(tables[table_name])
                       for table_name in tables}
        self.nobs = len(self.endog_inds)

        self.endog_names = [
            c for table_name in self.endog_inds.columns
            for c in self.tables[table_name]]
        self.exog_names = [
            c for table_name in self.exog_inds.columns
            for c in self.tables[table_name]]
        self.endog_colinds = np.cumsum(
            [0] + [tables[table_name].shape[1]
                   for table_name in self.endog_inds.columns])
        self.exog_colinds = np.cumsum(
            [0] + [tables[table_name].shape[1]
                   for table_name in self.exog_inds.columns])

        self.endog_shape = (self.nobs, self.endog_colinds[-1])
        self.exog_shape = (self.nobs, self.exog_colinds[-1])

        self.endog_scratch = np.empty((max_rows, self.endog_shape[1]))
        self.exog_scratch = np.empty((max_rows, self.exog_shape[1]))

        if max_rows is None:
            max_rows = self.nobs
        self.max_rows = max_rows

        assert len(self.endog_inds) == len(self.exog_inds)
        assert self.endog_shape[1] and self.exog_shape[1]
        assert len(self.endog_names) == len(set(self.endog_names))
        assert len(self.exog_names) == len(set(self.exog_names))

    def xnames(self):
        return self.exog_names

    def ynames(self):
        return self.endog_names

    def get_endog_exog(self, start, stop):
        for i in range(len(self.endog_inds.columns)):
            table_name = self.endog_inds.columns[i]
            self.endog_scratch[
                :stop - start,
                self.endog_colinds[i]:self.endog_colinds[i + 1]] = \
                self.tables[table_name].iloc[
                    self.endog_inds[table_name][start:stop, :]]
        for i in range(len(self.exog_inds.columns)):
            table_name = self.exog_inds.columns[i]
            self.exog_scratch[
                :stop - start,
                self.exog_colinds[i]:self.exog_colinds[i + 1]] = \
                self.tables[table_name].iloc[
                    self.exog_inds[table_name][start:stop, :]]
        return (self.endog_scratch[:stop - start, :],
                self.exog_scratch[:stop - start, :])

