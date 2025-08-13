#!/usr/bin/env python3
import argparse
import numpy as np


def generate_tcsm_sequence(steps: int, rho: float, init: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    g = np.zeros(steps, dtype=np.float64)
    prev = float(init)
    for t in range(steps):
        u = rng.random()
        gt = rho * prev + (1.0 - rho) * u
        # convex combination keeps it in [0,1] if prev,u in [0,1]
        gt = min(max(gt, 0.0), 1.0)
        g[t] = gt
        prev = gt
    return g


def generate_iid_sequence(steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.random(steps, dtype=np.float64)


def lag1_autocorr(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    if np.std(x0) == 0 or np.std(x1) == 0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def summarize(seq: np.ndarray, name: str):
    mean = float(np.mean(seq))
    var = float(np.var(seq))
    ac1 = lag1_autocorr(seq)
    print(f"{name}: steps={len(seq)} mean={mean:.4f} var={var:.6f} lag1_ac={ac1:.4f}")


def main():
    ap = argparse.ArgumentParser(description="TCSM gate sequence test")
    ap.add_argument("--steps", type=int, default=1000, help="number of steps to simulate")
    ap.add_argument("--rho", type=float, default=0.8, help="temporal correlation parameter in [0,1)")
    ap.add_argument("--init", type=float, default=0.5, help="initial gate value in [0,1]")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--compare_iid", action="store_true", help="also simulate IID gates for comparison")
    args = ap.parse_args()

    tcsm = generate_tcsm_sequence(args.steps, args.rho, args.init, args.seed)
    summarize(tcsm, name=f"TCSM(rho={args.rho:.2f})")

    if args.compare_iid:
        iid = generate_iid_sequence(args.steps, args.seed)
        summarize(iid, name="IID")

    # Optional: print first few values for sanity
    print("first_10:", np.array2string(tcsm[:10], precision=4, separator=","))


if __name__ == "__main__":
    main()
