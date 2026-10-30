"""트리거 탐지 알고리즘 검증 (가상 환자 시뮬레이션).

실행:  python validation/run_validation.py            (기본 200명, 약 1~2분)
       python validation/run_validation.py --n 50     (빠른 확인)

방법
  1) 가상 환자 N명 생성. 85%는 정답 트리거 1~2개(포드맵 하위그룹 + 30% 확률로 매운맛 추가),
     15%는 트리거가 없는 '대조 환자'(배경 증상만 있음).
  2) 기록 기간(7~56일)별로 앞부분 데이터만 잘라 분석.
  3) '의심 높음' 판정을 양성 예측으로 보고 정답과 비교:
       - 민감도  = 맞힌 정답 트리거 / 전체 정답 트리거
       - 정밀도  = 맞힌 예측 / 전체 '의심 높음' 예측
       - 환자당 거짓 경보 수
       - 1순위 적중률: 우리 방법 vs 단순 빈도법('증상 있던 식사에 가장 자주 들어있던 성분')
결과: validation/results.csv, validation/by_feature.csv, validation/summary.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fodmap import analysis, db  # noqa: E402
from fodmap.simulate import simulate_patient  # noqa: E402

DURATIONS = [7, 14, 21, 28, 42, 56]
CANDIDATES = db.SUBGROUPS + db.FLAGS
BETA = 2.2
START = date(2026, 10, 1)


def run(n: int = 200, seed: int = 2026, lag=(1, 8), threshold=4, min_n=5, verbose=True):
    ing, rec = db.load_ingredients(), db.load_recipes()
    lk = db.build_lookup(ing)
    rng = np.random.default_rng(seed)
    rows, feat_rows = [], []
    for i in range(n):
        if rng.random() < 0.15:
            truth = []
        else:
            k = 1 if rng.random() < 0.6 else 2
            truth = list(rng.choice(db.SUBGROUPS, size=k, replace=False))
            if rng.random() < 0.3:
                truth.append("caps")
        sim = simulate_patient(days=max(DURATIONS), triggers={t: BETA for t in truth}, seed=10_000 + i,
                               female=bool(rng.random() < 0.5), start=START, ing=ing, recipes=rec, lookup=lk)
        feats = analysis.add_confounders(analysis.meal_features(sim["meal_items"], ing, lk), sim["daily"])
        for D in DURATIONS:
            cut = pd.Timestamp(START + timedelta(days=D))
            f = feats[feats["datetime"] < cut].copy()
            s = sim["symptoms"][sim["symptoms"]["datetime"] < cut]
            f["outcome"] = analysis.label_outcomes(f, s, *lag, threshold)
            tt = analysis.trigger_table(f, min_n=min_n, seed=i)                      # v2(최종)
            tt1 = analysis.trigger_table(f, min_n=min_n, seed=i, stepwise=False, p_high=0.95)  # v1(초기)
            pred = set(tt.loc[tt["판정"] == "의심 높음", "feature"])
            pred1 = set(tt1.loc[tt1["판정"] == "의심 높음", "feature"])
            evaluable = tt[~tt["판정"].isin(["데이터 부족", "비교 불가(거의 매번 섭취)"])]
            top = evaluable["feature"].iloc[0] if len(evaluable) else None
            naive = analysis.naive_top(f)
            rows.append({
                "patient": i, "days": D, "n_true": len(truth), "tp": len(pred & set(truth)),
                "n_pred": len(pred), "fp": len(pred - set(truth)),
                "top_hit": (top in truth) if truth else np.nan,
                "naive_hit": (naive in truth) if truth else np.nan,
                "control": not truth,
                "tp_v1": len(pred1 & set(truth)), "n_pred_v1": len(pred1), "fp_v1": len(pred1 - set(truth)),
            })
            for t in truth:
                lab = tt.set_index("feature").loc[t, "판정"]
                feat_rows.append({"patient": i, "days": D, "feature": t, "detected": t in pred, "label": lab})
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n} patients")
    return pd.DataFrame(rows), pd.DataFrame(feat_rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("days")
    out = pd.DataFrame({
        "민감도": g["tp"].sum() / g["n_true"].sum(),
        "정밀도": g["tp"].sum() / g["n_pred"].sum().replace(0, np.nan),
        "환자당 거짓경보": g["fp"].mean(),
        "대조환자 거짓경보율": res[res["control"]].groupby("days")["n_pred"].apply(lambda x: (x > 0).mean()),
        "1순위 적중률(본 방법)": g["top_hit"].mean(),
        "v1 민감도": g["tp_v1"].sum() / g["n_true"].sum(),
        "v1 정밀도": g["tp_v1"].sum() / g["n_pred_v1"].sum().replace(0, np.nan),
        "v1 환자당 거짓경보": g["fp_v1"].mean(),
        "v1 대조환자 거짓경보율": res[res["control"]].groupby("days")["n_pred_v1"].apply(lambda x: (x > 0).mean()),
        "1순위 적중률(단순 빈도법)": g["naive_hit"].mean(),
    })
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    print(f"가상 환자 {args.n}명 시뮬레이션 중...")
    res, feat = run(args.n)
    summ = summarize(res)
    byf = feat.groupby(["feature", "days"])["detected"].mean().unstack("days")
    byf.index = [db.FEATURE_KO[i] for i in byf.index]
    unc = feat[feat["days"] == 28].groupby("feature")["label"].apply(
        lambda x: (x == "비교 불가(거의 매번 섭취)").mean())
    out = Path(__file__).parent
    summ.to_csv(out / "results.csv", index=False)
    byf.to_csv(out / "by_feature.csv")
    md = ["# 트리거 탐지 검증 결과", "",
          f"- 가상 환자 {args.n}명 (대조 환자 {int(res[res.days == 7].control.sum())}명 포함), 트리거 효과 β={BETA}",
          "- v2(최종): P(노출 시 증상률 > 비노출 시 증상률) ≥ 0.99 + 단계적 조건부 검정",
          "- v1(초기): 같은 확률 ≥ 0.95, 각 요인을 따로 검정 (다중비교·동반섭취로 거짓경보 많음)", "",
          "## 기록 기간별 성능", "", summ.round(3).to_markdown(index=False), "",
          "## 성분별 민감도 (기록 기간별)", "", byf.round(2).to_markdown(), "",
          "## 28일 시점 '비교 불가(거의 매번 섭취)' 비율", "",
          unc.rename(index=db.FEATURE_KO).round(2).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summ.round(3).to_string(index=False))
    print(byf.round(2).to_string())


if __name__ == "__main__":
    main()
