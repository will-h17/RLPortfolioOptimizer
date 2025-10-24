from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple, Optional
import pandas as pd

@dataclass(frozen=True)
class TimeSplit:
    train: slice
    val: slice
    test: slice

def date_slices(index: pd.DatetimeIndex,
                train_end: str,
                val_end: str,
                test_end: str) -> TimeSplit:
    """Return integer slices for train/val/test using *dates* (inclusive boundaries)."""
    def pos(d): 
        return int(index.get_indexer([pd.to_datetime(d)], method="bfill")[0])
    i_train_end = pos(train_end)
    i_val_end   = pos(val_end)
    i_test_end  = pos(test_end)
    return TimeSplit(
        train=slice(0, i_train_end+1),
        val=slice(i_train_end+1, i_val_end+1),
        test=slice(i_val_end+1, i_test_end+1)
    )

def walk_forward(index: pd.DatetimeIndex,
                 train_len: int,
                 val_len: int,
                 test_len: int,
                 step: Optional[int]=None) -> Iterator[TimeSplit]:
    """
    Rolling windows in *rows* (bars). Yields TimeSplit with integer slices.
    """
    n = len(index)
    step = step or test_len
    start = 0
    while True:
        tr0 = start
        tr1 = tr0 + train_len
        va1 = tr1 + val_len
        te1 = va1 + test_len
        if te1 > n: break
        yield TimeSplit(slice(tr0, tr1), slice(tr1, va1), slice(va1, te1))
        start += step
