import abc
import numpy as np
import pandas as pd


def to_dataframe(data):
    """Sets df = pd.DataFrame(data), and then cleans up df.columns so it
    is a list of unique strings (instead of a MultiIndex, being a generator,
    containing ints or something else, or having duplicate column names).
    """
    df = pd.DataFrame(data)
    if isinstance(df.columns, pd.MultiIndex):
        cols = ["_".join(filter(None, map(str, c))) for c in df.columns]
    else:
        cols = list(map(str, df.columns))
    counts = {c: (n, 0) for c, n in zip(*np.unique(cols, return_counts=True))}
    for i in range(len(cols)):
        c = cols[i]
        if counts[c][0] > 1:
            cols[i] = c + "_" + str(counts[c][1])
            counts[c][1] += 1
    df.columns = cols
    return df


def yx_to_dataframe(data):
    """Options for data:
    - data is tuple containing (endog, exog): In this case, we initialize
      endog and exog using to_dataframe. If the resulting column names
      overlap, we prefix the exog column names with "x" and the endog
      column names with "y". For example, if plain numpy arrays are given,
      the column names end up being ["y0", "y1", ... , "x0", "x1", ...].
    - data is anything else: We then initialize it by passing it through
      to_dataframe.
    """
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("data is tuple not of length 2")
        endog = to_dataframe(data[0])
        exog = to_dataframe(data[1])
        df = pd.concat([endog, exog], axis=1, copy=False)
        if set(endog.columns) & set(exog.columns):
            ycols = ["y" + str(c) for c in endog.columns]
            xcols = ["x" + str(c) for c in exog.columns]
            df.columns = ycols + xcols
        return df
    return to_dataframe(data)


class AbstractModelData(abc.ABC):
    shape = NotImplemented
    nobs = NotImplemented
    columns = NotImplemented

    # this is the max # of rows that can be requested through get_data
    max_rows = NotImplemented

    @abc.abstractmethod
    def get_data(self, start, stop):
        """Returns the rows indexed by start:stop, as ndarray. It is assumed
        that 0 <= start, stop <= self.nobs and stop - start <= max_rows.

        Ideally this function should not allocate memory, and to allow that it
        can invalidate previous return values.

        Also note that this will only be called (by the model) in a sequence
        resulting in a sweep over the data (e.g. 0:100, 100:200, 200:300, etc),
        possibly going through multiple times.
        """
        NotImplemented

    @abc.abstractmethod
    def subdata(self, colnames):
        """Basically returns data.loc[:, colnames] as AbstractModelData, except
        the order of columns in the result does not have to be the same.
        """
        NotImplemented

    def _col_to_i(self):
        return {self.columns[i]: i for i in range(len(self.columns))}

    def _sorted_colnames(self, colnames, col_to_i=None):
        col_to_i = self._col_to_i() if col_to_i is None else col_to_i
        return sorted(map(str, colnames), key=lambda c: col_to_i[c])


class SimpleModelData(AbstractModelData):
    def __init__(self, data, max_rows=None):
        """data is passed through yx_to_dataframe"""
        self.data = yx_to_dataframe(data)
        if not self.data.size:
            raise ValueError("data is empty")

        self.shape = self.data.shape
        self.columns = self.data.columns
        self.nobs = len(self.data)
        if max_rows is None:
            max_rows = len(self.data)
        self.max_rows = max_rows

        self.scratch = np.empty((max_rows, self.shape[1]))

    def get_data(self, start, stop):
        self.scratch[: stop - start, :] = self.data.iloc[start:stop]
        return self.scratch[: stop - start, :]

    def subdata(self, colnames):
        colnames_l = self._sorted_colnames(colnames)
        if list(self.columns) == colnames_l:
            return self
        return SimpleModelData(self.data.loc[:, colnames_l], self.max_rows)


class IndexedModelData(AbstractModelData):
    """
    Here,
    actual_data = concatenate(
        [
            tables[table_name].iloc[data_inds[table_name], :]
            for table_name in data_inds
        ],
        axis=1,
    )
    """

    def __init__(self, data_inds, tables, max_rows=None):
        """
        Note that tables, as passed in, needs to be a dict-like with values
        that can be passed into to_dataframe.

        data_inds will be passed through yx_to_dataframe.
        """
        self.data_inds = yx_to_dataframe(data_inds)
        if not self.data_inds.size:
            raise ValueError("data_inds is empty")
        tables_dict = dict(tables)
        self.tables = []
        for table_name in self.data_inds.columns:
            if table_name not in tables_dict:
                raise ValueError(f"Missing '{table_name}' in tables")
            df = to_dataframe(tables_dict[table_name])
            self.tables.append(df)
            col = self.data_inds[table_name]
            if np.min(col) < -df.shape[0] or np.max(col) >= df.shape[0]:
                msg = f"Out of bounds indices in data_inds[{table_name}]"
                raise ValueError(msg)

        table_widths = [df.shape[1] for df in self.tables]
        self.colinds = np.cumsum([0] + table_widths)

        self.nobs = len(self.data_inds)
        self.shape = (self.nobs, self.colinds[-1])
        self.columns = [c for df in self.tables for c in df.columns]
        if max_rows is None:
            max_rows = self.nobs
        self.max_rows = max_rows

        self.scratch = np.empty((self.max_rows, self.shape[1]))

    def get_data(self, start, stop):
        for i in range(len(self.tables)):
            df = self.tables[i]
            table_name = self.endog_inds.columns[i]
            self.scratch[
                : stop - start,
                self.colinds[i] : self.colinds[i + 1],
            ] = df.iloc[self.endog_inds[table_name].iloc[start:stop]]
        return self.scratch[: stop - start, :]

    def subdata(self, colnames):
        col_to_i = self._col_to_i()
        colnames_l = self._sorted_colnames(colnames, col_to_i)
        if self.columns == colnames_l:
            return self
        data_inds = []
        tables = {}
        ti = -1
        for c in colnames_l:
            ci = col_to_i[c]
            old_ti = ti
            while ci >= self.colinds[ti + 1]:
                ti += 1
            table_name = self.data_inds.columns[ti]
            if ti != old_ti:
                data_inds.append(self.data_inds.iloc[:, ti])
                tables[table_name] = []
            tables[table_name].append(self.tables[ti][c])
        return IndexedModelData(
            pd.concat(data_inds, axis=1, copy=False),
            {t: pd.concat(l, axis=1, copy=False) for t, l in tables.items()},
        )
