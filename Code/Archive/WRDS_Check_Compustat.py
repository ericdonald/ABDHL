"""Archived 2026-11-18: one-off comparison of a new Compustat pull with the file it replaced (was called by Processor.Cleaner before overwriting compustat.csv; prints only)."""
import numpy as np
import pandas as pd


def Check_Compustat(new, old_path):
    "Before compustat.csv is overwritten: how the new pull compares with the file it replaces (rows, firms, matching values)."
    if not old_path.exists():
        return
    old = pd.read_csv(old_path, low_memory=False)
    new = new.assign(gvkey=pd.to_numeric(new.gvkey))
    shared = [c for c in new.columns if c in old.columns]
    print(f'Compustat check: new {len(new):,} rows / {new.gvkey.nunique():,} firms; existing file {len(old):,} rows / {old.gvkey.nunique():,} firms; '
          f'columns only in the existing file: {sorted(set(old.columns) - set(new.columns))}', flush=True)
    m = old.merge(new, on=['gvkey', 'fyear'], suffixes=('_old', '_new'))
    print(f'  firm-years in both: {len(m):,} (existing file {len(m) / len(old):.1%}, new pull {len(m) / len(new):.1%})', flush=True)
    for c in [c for c in shared if c not in ('gvkey', 'fyear')]:
        a, b = m[c + '_old'], m[c + '_new']
        if a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
            same = (np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, equal_nan=True)).mean()
        else:
            same = (a.astype(str) == b.astype(str)).mean()
        print(f'  {c:8s} identical in {same:.2%} of common firm-years', flush=True)
