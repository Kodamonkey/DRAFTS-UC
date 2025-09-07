from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import pandas as pd


PRECISE_FILES = [
    'Results/3096_0001_00_8bit.candidates_with_mjd_bary_precise.csv',
    'Results/3098_0001_00_8bit.candidates_with_mjd_bary_precise.csv',
    'Results/3099_0001_00_8bit.candidates_with_mjd_bary_precise.csv',
    'Results/3100_0001_00_8bit.candidates_with_mjd_bary_precise.csv',
    'Results/3101_0001_00_8bit.candidates_with_mjd_bary_precise.csv',
    'Results/3102_0001_00_8bit.candidates_with_mjd_bary_precise.csv',
]

BURSTS_CSV = 'scripts/bursts_mjd.csv'

OUT_DIR = Path('Results')
OUT_CAND = OUT_DIR / 'summary_candidates_mjd.csv'
OUT_BURSTS = OUT_DIR / 'summary_bursts_mjd.csv'
OUT_MATCH = OUT_DIR / 'summary_matches.csv'


def load_candidates() -> pd.DataFrame:
    rows = []
    for path in PRECISE_FILES:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if 'mjd_bary_utc_inf' not in df.columns:
            continue
        base = p.name.split('.')[0]
        # index per row for reproducibility
        for idx, r in df.iterrows():
            rows.append({
                'file': r.get('file', base),
                'row_idx': int(idx),
                'mjd_bary_utc_inf': float(r['mjd_bary_utc_inf']),
                'mjd_day': int(float(r['mjd_bary_utc_inf'])),
                't_sec_centered': float(r.get('t_sec_centered', r.get('t_sec', 0.0))),
                'dm_pc_cm-3': float(r.get('dm_pc_cm-3', np.nan)),
            })
    return pd.DataFrame(rows)


def load_bursts() -> pd.DataFrame:
    df = pd.read_csv(BURSTS_CSV)
    df = df.rename(columns={'Label': 'label', 'MJD': 'mjd'})
    df['mjd'] = pd.to_numeric(df['mjd'], errors='coerce')
    df = df.dropna(subset=['mjd'])
    df['mjd_day'] = df['mjd'].astype(float).apply(lambda v: int(v))
    return df[['label', 'mjd', 'mjd_day']]


def build_matches(cand: pd.DataFrame, bursts: pd.DataFrame) -> pd.DataFrame:
    # Group bursts by day for faster lookup
    bursts_by_day = {d: g[['label', 'mjd']].to_numpy() for d, g in bursts.groupby('mjd_day')}
    all_bursts = bursts[['label', 'mjd']].to_numpy()
    match_rows = []
    for i, r in cand.iterrows():
        xi = float(r['mjd_bary_utc_inf'])
        day = int(r['mjd_day'])
        # Prefer same day; fallback to all
        pool = bursts_by_day.get(day)
        if pool is None or len(pool) == 0:
            pool = all_bursts
        # Find closest
        best_label, best_mjd, best_dt = None, None, None
        for lab, mjd in pool:
            dt = abs(float(mjd) - xi)
            if best_dt is None or dt < best_dt:
                best_label, best_mjd, best_dt = lab, float(mjd), dt
        match_rows.append({
            'file': r['file'],
            'row_idx': r['row_idx'],
            'candidate_mjd': xi,
            'candidate_mjd_day': day,
            'burst_label': best_label,
            'burst_mjd': best_mjd,
            'burst_mjd_day': int(best_mjd),
            'delta_s': float(best_dt * 86400.0),
        })
    return pd.DataFrame(match_rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand = load_candidates()
    bursts = load_bursts()
    cand.to_csv(OUT_CAND, index=False)
    bursts.to_csv(OUT_BURSTS, index=False)
    matches = build_matches(cand, bursts)
    matches.sort_values(['file', 'delta_s'], inplace=True)
    matches.to_csv(OUT_MATCH, index=False)
    print(f'✅ Escrito {OUT_CAND}')
    print(f'✅ Escrito {OUT_BURSTS}')
    print(f'✅ Escrito {OUT_MATCH}')


if __name__ == '__main__':
    main()


