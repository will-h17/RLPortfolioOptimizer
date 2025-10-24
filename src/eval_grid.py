from __future__ import annotations
import argparse, json, re, glob, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.scaler import FitOnTrainScaler
from src.repro import set_global_seed
from src.backtest import evaluate_sb3_model  # uses your PortfolioEnv properly
from src.baselines import (
    CostCfg,
    vt_equal_weight,
    ivp_inverse_variance,
    momentum_xs,
    _run_backtest,
)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# ---------- helpers ----------

def _auto_latest_run(artifacts_root: Path) -> Path:
    runs = sorted(artifacts_root.glob("*__*"), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("No run folders in artifacts/. Train first.")
    return runs[-1]

def _load_cfg(p: str | Path) -> dict:
    p = Path(p)
    if p.suffix in {".yml", ".yaml"}:
        import yaml
        return yaml.safe_load(p.read_text())
    return json.loads(p.read_text())

def _select_universe(prices: pd.DataFrame, tickers) -> pd.DataFrame:
    if tickers == "*" or tickers == ["*"]:
        return prices
    cols = [c for c in prices.columns if c in set(tickers)]
    if not cols:
        raise ValueError(f"No matching tickers in prices for {tickers}")
    return prices[cols]

def _slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    mask = (df.index >= s) & (df.index <= e)
    out = df.loc[mask]
    return out

def _filter_features(feats: pd.DataFrame, keep_pat: str) -> pd.DataFrame:
    if keep_pat in ("*", "", None):
        return feats
    pat = re.compile(keep_pat)
    keep = [c for c in feats.columns if pat.search(c)]
    if not keep:
        raise ValueError(f"feature_keep='{keep_pat}' kept zero columns")
    return feats[keep]

def _align(prices: pd.DataFrame, feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feats = feats.dropna(how="any")
    idx = prices.index.intersection(feats.index)
    if len(idx) == 0:
        raise ValueError("No common timestamps after dropna(features).")
    return prices.loc[idx], feats.loc[idx]

def _eval_rl(P: pd.DataFrame, Xz: pd.DataFrame, model_zip: Path) -> Dict[str, float]:
    res = evaluate_sb3_model(P, Xz, str(model_zip))
    m = res["metrics"]
    return dict(sharpe=m["sharpe"], max_drawdown=m["max_drawdown"], turnover=m["turnover"], steps=len(res["equity_curve"]))

def _eval_baselines(P: pd.DataFrame, feats_raw: Optional[pd.DataFrame], ab_cfg: dict, bl_cfg: dict) -> List[Dict[str, object]]:
    rows = []
    # cost: use flat_fraction when enabled=False (simpler for ablations)
    flat = float(ab_cfg.get("cost_bps", 0)) / 10_000.0
    cost_cfg = CostCfg(enabled=False, flat_fraction=flat)

    if bl_cfg.get("vt_ew", {}).get("enabled", True):
        t = vt_equal_weight(
            P,
            window=bl_cfg["vt_ew"].get("window", 60),
            target_vol=bl_cfg["vt_ew"].get("target_vol", 0.10),
            lev_cap=1.0,
            rebalance=bl_cfg["vt_ew"].get("rebalance", "W-FRI"),
        )
        r = _run_backtest(P, t, cost_cfg, features=None, band=0.0025)
        rows.append(("VT-EW", r))
    if bl_cfg.get("ivp", {}).get("enabled", True):
        t = ivp_inverse_variance(
            P,
            window=bl_cfg["ivp"].get("window", 60),
            rebalance=bl_cfg["ivp"].get("rebalance", "W-FRI"),
        )
        r = _run_backtest(P, t, cost_cfg, features=None, band=0.0025)
        rows.append(("IVP", r))
    if bl_cfg.get("momentum", {}).get("enabled", True):
        t = momentum_xs(
            P,
            lookback_days=bl_cfg["momentum"].get("lookback_days", 252),
            skip_days=bl_cfg["momentum"].get("skip_days", 21),
            top_frac=bl_cfg["momentum"].get("top_frac", 0.3),
            rebalance=bl_cfg["momentum"].get("rebalance", "M"),
        )
        r = _run_backtest(P, t, cost_cfg, features=None, band=0.0)
        rows.append(("MOM", r))

    out = []
    for name, r in rows:
        m = r["metrics"]
        out.append(dict(model=name, sharpe=m["sharpe"], max_drawdown=m["max_drawdown"], turnover=m["turnover"], steps=m["num_steps"]))
    return out

def _align_with_warmup(prices: pd.DataFrame, feats: pd.DataFrame, min_rows:int=20):
    """
    Aligns P and F after dropping feature warm-up NaNs.
    If the selected period has only NaNs in features, return (None, None, reason).
    Also clamps start to the first fully-valid feature row.
    """
    # drop rows where ANY feature is NaN
    F2 = feats.dropna(how="any")
    if F2.empty:
        return None, None, "all features NaN in this slice (period too early for lookback)"
    first_ok = F2.index.min()
    # clamp both to start at first_ok
    P2 = prices.loc[prices.index >= first_ok]
    F2 = F2.loc[F2.index >= first_ok]
    # final inner join
    idx = P2.index.intersection(F2.index)
    if len(idx) < min_rows:
        return None, None, f"only {len(idx)} usable rows after warm-up; need >= {min_rows}"
    return P2.loc[idx], F2.loc[idx], None


# ---------- main grid ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="Path to eval_grid.yaml")
    args = ap.parse_args()

    grid = _load_cfg(args.grid)

    # Locate run & artifacts (model + scaler)
    artifacts = ROOT / "artifacts"
    run_dir = Path(grid.get("run_dir") or _auto_latest_run(artifacts))
    scaler_path = run_dir / "scaler.json"
    best_zip = run_dir / "models" / "best" / "best_model.zip"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.json at {scaler_path}")
    model_zip = best_zip if best_zip.exists() else (run_dir / "models" / "ppo_dirichlet_sb3.zip")

    # Global seed for determinism
    set_global_seed( int(grid.get("seed", 1337)) , deterministic=True)

    # Load data
    prices_all = pd.read_parquet(PROC / "prices_adj.parquet")
    feats_all  = pd.read_parquet(PROC / "features.parquet")

    # Results accumulator
    rows: List[Dict[str, object]] = []

    # Iterate scenarios
    for uname, udef in grid["universes"].items():
        P_u = _select_universe(prices_all, udef.get("tickers", "*"))

        for pname, pdef in grid["periods"].items():
            P_up = _slice_period(P_u, pdef["start"], pdef["end"])
            F_up = _slice_period(feats_all, pdef["start"], pdef["end"])

            # Align and drop NaNs
            res = _align_with_warmup(P_up, F_up, min_rows=20)
            if res[2] is not None:
                print(f"[skip] {uname}/{pname}: {res[2]}")
                continue
            P_up, F_up, _ = res


            for ab in grid["ablations"]:
                ab_name = ab["name"]

                # Feature selection (raw), then align to prices again
                F_sel = _filter_features(F_up, ab.get("feature_keep", "*"))
                res2 = _align_with_warmup(P_up, F_sel, min_rows=20)
                if res2[2] is not None:
                    print(f"[skip] {uname}/{pname}/{ab_name}: {res2[2]}")
                    continue
                P_s, F_s, _ = res2

                # Load scaler fitted on TRAIN of the run, transform features of this slice
                scaler = FitOnTrainScaler.load(scaler_path, columns=feats_all.columns)
                # determine scaler column names with several fallbacks for compatibility
                scaler_cols = getattr(scaler, "columns_", None)
                if scaler_cols is None:
                    scaler_cols = getattr(scaler, "columns", None)
                if scaler_cols is None:
                    scaler_cols = getattr(scaler, "feature_names_in_", None)
                if scaler_cols is None:
                    scaler_cols = []
                scaler_cols = list(scaler_cols)
                # keep only selected feature columns for transform (intersection with scaler columns)
                keep_cols = [c for c in F_s.columns if c in set(scaler_cols)]
                X_s = F_s[keep_cols]
                X_sz = scaler.transform(X_s)

                # Align transformed features to prices (defensive)
                common = P_s.index.intersection(X_sz.index)
                P_s = P_s.loc[common]
                X_sz = X_sz.loc[common]

                # --- RL eval ---
                try:
                    m_rl = _eval_rl(P_s, X_sz, model_zip)
                    rows.append(dict(model="RL", universe=uname, period=pname, ablation=ab_name, **m_rl))
                except Exception as e:
                    rows.append(dict(model="RL", universe=uname, period=pname, ablation=ab_name, error=f"{type(e).__name__}: {e}"))

                # --- Baselines ---
                base_rows = _eval_baselines(P_s, feats_raw=None, ab_cfg=ab, bl_cfg=grid.get("baselines", {}))
                for br in base_rows:
                    rows.append(dict(universe=uname, period=pname, ablation=ab_name, **br))

    # Save outputs
    out_dir = run_dir / "eval" / "breadth"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "grid_results.csv", index=False)

    # quick summaries
    if not df.empty and "error" in df.columns:
        errs = df[df["error"].notna()]
        if not errs.empty:
            errs.to_csv(out_dir / "errors.csv", index=False)

    # model pivot: Sharpe median per (model, universe, period, ablation)
    if not df.empty:
        piv = (df.dropna(subset=["sharpe"])
                 .groupby(["model","universe","period","ablation"], as_index=False)["sharpe"]
                 .median()
               )
        piv.to_csv(out_dir / "grid_sharpe_median.csv", index=False)

    print(f"[done] Wrote {len(df)} rows to {out_dir/'grid_results.csv'}")

if __name__ == "__main__":
    main()
