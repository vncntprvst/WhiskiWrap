"""Cross-session validation of the learned add-on (sc013 <-> seg04, different animals).

Turns the generalization claim from asserted to measured:
- COVERAGE (claimed generalizable): train on clip A, evaluate real-vs-noise AP on clip B.
- END-TO-END on a NEW clip with the coverage model trained ELSEWHERE + per-clip identity
  (bootstrap = no labels, the realistic new-clip case; and GT-trained = upper bound).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import autotune_linking as at
from . import benchmark_linking as bl
from . import coverage_model as cov
from . import identity_model as idm
from . import eval_linking as ev
from .gt_labels import match_to_gt

WIN = {"coverage": "model", "identity": "rerank", "keep": 0.4, "w_model": 1.2}


def cov_ap(train_comb, train_gt, test_comb, test_gt) -> float:
    b = cov.train_coverage(train_comb, train_gt)          # all frames of the train clip
    test = match_to_gt(test_comb, test_gt)
    return average_precision_score(test["is_real"].astype(int), cov.predict_real(test, b))


def e2e(cache, gt, cov_bundle, id_bundle, cfg=WIN) -> dict:
    o = ev.compute_metrics(at.apply_config(cache, cfg, cov_bundle, id_bundle), gt)["overall"]
    return {k: round(o[k], 4) if isinstance(o[k], float) else o[k]
            for k in ["identity_accuracy", "min_coverage", "miss", "total_id_switches", "idf1", "false_positive"]}


def main():
    import warnings; warnings.filterwarnings("ignore")
    print("caching backbones (sc013 + seg04)...")
    sc = at.cache_backbone("sc013_active")
    seg = at.cache_backbone("seg04")
    gts = pd.read_parquet(bl.CLIPS["sc013_active"]["gt"])
    gtg = pd.read_parquet(bl.CLIPS["seg04"]["gt"])

    print("\n=== 1. COVERAGE real-vs-noise AP (cross-session) ===")
    print(f"  train sc013 -> test seg04 : AP={cov_ap(sc['combined'], gts, seg['combined'], gtg):.4f}")
    print(f"  train seg04 -> test sc013 : AP={cov_ap(seg['combined'], gtg, sc['combined'], gts):.4f}")

    cov_sc = cov.train_coverage(sc["combined"], gts)      # trained on sc013 ONLY
    cov_sg = cov.train_coverage(seg["combined"], gtg)     # trained on seg04 ONLY

    print("\n=== 2. END-TO-END on seg04 (NEW animal); coverage trained on sc013 ===")
    print(f"  baseline (filters)            : {e2e(seg, gtg, None, None, {'coverage':'filters','identity':'off'})}")
    id_boot = idm.train_identity(seg["out"], gt=None)     # bootstrap, NO seg04 labels
    print(f"  add-on cov=sc013 id=bootstrap : {e2e(seg, gtg, cov_sc, id_boot)}")
    id_gtg = idm.train_identity(gtg, gt=gtg)              # per-session GT (upper bound)
    print(f"  add-on cov=sc013 id=seg04GT   : {e2e(seg, gtg, cov_sc, id_gtg)}")

    print("\n=== 3. END-TO-END on sc013; coverage trained on seg04 (reverse) ===")
    print(f"  baseline (filters)            : {e2e(sc, gts, None, None, {'coverage':'filters','identity':'off'})}")
    id_sc = idm.train_identity(gts, gt=gts)
    print(f"  add-on cov=seg04 id=sc013GT   : {e2e(sc, gts, cov_sg, id_sc)}")


if __name__ == "__main__":
    main()
