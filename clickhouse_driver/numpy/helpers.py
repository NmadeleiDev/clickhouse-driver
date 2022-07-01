import numpy as np
import pandas as pd


def column_chunks(columns, n):
    columns_casted = []
    for column in columns:
        if not isinstance(column, (np.ndarray, pd.DatetimeIndex)):
            casted = None
            if hasattr(column, 'to_numpy') and callable(column.to_numpy):
                casted = column.to_numpy()
            else:
                try:
                    casted = np.array(column)
                except Exception as e:
                    raise TypeError(f'Failed to call np.array(column) on colume type = {type(column)}, sample: {column[:20]}, exception: {e}')
            if casted is None:
                raise TypeError(
                    'Unsupported column type: {}. '
                    'ndarray/DatetimeIndex is expected. Col sample: {}'
                    .format(type(column), column[:20])
                )
            columns_casted.append(casted)
        else:
            columns_casted.append(column)

    columns = columns_casted

    # create chunk generator for every column
    chunked = [
        iter(np.array_split(c, len(c) // n) if len(c) > n else [c])
        for c in columns
    ]

    while True:
        # get next chunk for every column
        item = [next(column, []) for column in chunked]
        if not any(len(x) for x in item):
            break
        yield item
