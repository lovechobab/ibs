"""개인 트리거 분석.

핵심 아이디어
-------------
각 '식사'를 하나의 관찰 단위로 본다.
  - 노출(exposure): 그 식사의 하위그룹 부하 ≥ 0.5 이면 '노출', 포드맵 외 요인(매운맛 등)은 포함 여부
  - 결과(outcome): 식후 [lag_min, lag_max]시간 안에 증상 점수 ≥ threshold 기록이 있으면 1
노출된 식사와 노출되지 않은 식사의 증상 발생률을 베이지안(Beta-Binomial)으로 비교해
  - 위험비(RR)와 95% 신용구간
  - P(노출 시 증상률 > 비노출 시 증상률)
을 계산한다. 연관성이지 인과관계가 아니므로 최종 확인은 재도입 챌린지로 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .db import EXPOSE_LOAD, FEATURE_KO, FLAGS, SUBGROUPS, assess_meal, build_lookup

SYMPTOM_COLS = ["pain", "bloating", "diarrhea", "constipation"]
CONFOUNDERS = {"conf_stress": "스트레스 높음(7점 이상)", "conf_sleep": "수면 부족(6시간 미만)", "conf_mens": "월경 기간"}


def meal_features(meal_items: pd.DataFrame, ing: pd.DataFrame, lookup: dict | None = None) -> pd.DataFrame:
    lookup = lookup or build_lookup(ing)
    recs = []
    if meal_items is None or meal_items.empty:
        return pd.DataFrame()
    for mid, g in meal_items.groupby("meal_id", sort=False):
        a = assess_meal(list(zip(g["ingredient"], g["grams"])), lookup)
        rec = {
            "meal_id": mid,
            "datetime": pd.Timestamp(g["datetime"].iloc[0]),
            "meal_label": g["meal_label"].iloc[0],
            "overall": a["overall"],
            "ingredients": "|".join(sorted(set(g.loc[g["grams"] > 0, "ingredient"]))),
        }
        for s in SUBGROUPS:
            rec[f"load_{s}"] = a["loads"][s]
            rec[f"exp_{s}"] = a["loads"][s] >= EXPOSE_LOAD
        for f in FLAGS:
            rec[f"exp_{f}"] = f in a["flags"]
        recs.append(rec)
    return pd.DataFrame(recs).sort_values("datetime").reset_index(drop=True)


def symptom_scores(symptoms: pd.DataFrame) -> pd.DataFrame:
    s = symptoms.copy()
    s["datetime"] = pd.to_datetime(s["datetime"])
    s["score"] = s[SYMPTOM_COLS].astype(float).max(axis=1)
    return s


def label_outcomes(feats: pd.DataFrame, symptoms: pd.DataFrame, lag_min: float = 1, lag_max: float = 8,
                   threshold: float = 4) -> pd.Series:
    if feats.empty:
        return pd.Series(dtype=int)
    if symptoms is None or symptoms.empty:
        return pd.Series(0, index=feats.index)
    s = symptom_scores(symptoms)
    times = np.sort(s.loc[s["score"] >= threshold, "datetime"].values.astype("datetime64[ns]"))
    t = feats["datetime"].values.astype("datetime64[ns]")
    lo = t + np.timedelta64(int(lag_min * 3600), "s")
    hi = t + np.timedelta64(int(lag_max * 3600), "s")
    # (lo, hi] 구간에 증상 기록이 하나라도 있는가
    n_lo = np.searchsorted(times, lo, side="right")
    n_hi = np.searchsorted(times, hi, side="right")
    return pd.Series((n_hi > n_lo).astype(int), index=feats.index)


def add_confounders(feats: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    for c in CONFOUNDERS:
        out[c] = False
    if daily is None or daily.empty or out.empty:
        return out
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.drop_duplicates("date", keep="last").set_index("date")
    dates = out["datetime"].dt.date
    out["conf_stress"] = dates.map(d["stress"].astype(float) >= 7).fillna(False).astype(bool)
    out["conf_sleep"] = dates.map(d["sleep_h"].astype(float) < 6).fillna(False).astype(bool)
    out["conf_mens"] = dates.map(d["menstruation"].astype(bool)).fillna(False).astype(bool)
    return out


def build_dataset(meal_items, symptoms, daily, ing, lag_min=1, lag_max=8, threshold=4, lookup=None) -> pd.DataFrame:
    feats = meal_features(meal_items, ing, lookup)
    if feats.empty:
        return feats
    feats["outcome"] = label_outcomes(feats, symptoms, lag_min, lag_max, threshold)
    return add_confounders(feats, daily)


def bayes_compare(x, y, rng, draws: int = 4000, prior=(1.0, 1.0)) -> dict:
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=int)
    n1, n0 = int(x.sum()), int((~x).sum())
    k1, k0 = int(y[x].sum()), int(y[~x].sum())
    a, b = prior
    p1 = rng.beta(k1 + a, n1 - k1 + b, draws)
    p0 = rng.beta(k0 + a, n0 - k0 + b, draws)
    rr = p1 / p0
    return {
        "n1": n1, "k1": k1, "n0": n0, "k0": k0,
        "rate1": k1 / n1 if n1 else np.nan,
        "rate0": k0 / n0 if n0 else np.nan,
        "rr": float(np.median(rr)),
        "rr_lo": float(np.quantile(rr, 0.025)),
        "rr_hi": float(np.quantile(rr, 0.975)),
        "p_greater": float((p1 > p0).mean()),
    }


P_HIGH = 0.99   # 10개 요인을 동시에 검사하므로(다중비교) 0.95보다 엄격하게. 검증 결과로 선택.
P_MID = 0.80


def classify(res: dict, min_n: int = 5, p_high: float = P_HIGH) -> str:
    if res["n1"] < min_n:
        return "데이터 부족"
    if res["n0"] < min_n:
        return "비교 불가(거의 매번 섭취)"
    if res["p_greater"] >= p_high:
        return "의심 높음"
    if res["p_greater"] >= P_MID:
        return "가능성 있음"
    return "근거 약함"


LABEL_ORDER = {"의심 높음": 0, "가능성 있음": 1, "근거 약함": 2, "비교 불가(거의 매번 섭취)": 3, "데이터 부족": 4}


def _one_pass(df, features, min_n, rng, p_high=P_HIGH):
    rows = {}
    for f in features:
        col = f"exp_{f}"
        if col not in df:
            continue
        r = bayes_compare(df[col].values, df["outcome"].values, rng)
        r["feature"] = f
        r["판정"] = classify(r, min_n, p_high)
        rows[f] = r
    return rows


def trigger_table(df: pd.DataFrame, min_n: int = 5, seed: int = 0,
                  features: list[str] | None = None, stepwise: bool = True,
                  p_high: float = P_HIGH) -> pd.DataFrame:
    """stepwise=True: 단계적 조건부 검정
       1) 전체 식사로 비교 → '의심 높음' 중 가장 강한 요인 1개를 확정
       2) 확정된 요인이 '없는' 식사만 남겨 나머지 요인을 다시 비교 → 반복
       같은 음식에 함께 들어가는 성분(예: 김치찌개의 마늘+고춧가루)이 덩달아 의심받는 것을 줄인다."""
    rng = np.random.default_rng(seed)
    features = [f for f in (features or (SUBGROUPS + FLAGS)) if f"exp_{f}" in df]
    first = _one_pass(df, features, min_n, rng, p_high)
    final = {f: dict(r, 단계="전체 비교") for f, r in first.items()}
    if stepwise:
        confirmed, sub, remaining = [], df, list(features)
        while True:
            res = _one_pass(sub, remaining, min_n, rng, p_high)
            hits = [r for r in res.values() if r["판정"] == "의심 높음"]
            for f, r in res.items():
                if f not in confirmed and confirmed:
                    final[f] = dict(r, 단계=f"조건부 비교({', '.join(FEATURE_KO[c] for c in confirmed)} 제외)")
                    if r["판정"] in ("데이터 부족", "비교 불가(거의 매번 섭취)"):
                        # 조건부로는 판단 불가 → 전체 비교 결과를 쓰되 '의심 높음'은 '가능성 있음'으로 낮춤
                        fr = dict(first[f])
                        if fr["판정"] == "의심 높음":
                            fr["판정"] = "가능성 있음"
                        final[f] = dict(fr, 단계="전체 비교(조건부 비교 불가)")
            if not hits:
                break
            best = max(hits, key=lambda r: (r["p_greater"], r["rr"]))
            confirmed.append(best["feature"])
            final[best["feature"]] = dict(res[best["feature"]], 단계=final[best["feature"]]["단계"])
            remaining.remove(best["feature"])
            sub = sub[~sub[f"exp_{best['feature']}"]]
            if not remaining or len(sub) < 2 * min_n:
                for f in remaining:
                    if final[f]["판정"] == "의심 높음" and final[f]["단계"] == "전체 비교":
                        final[f]["판정"] = "가능성 있음"
                break
    rows = []
    for f in features:
        r = final[f]
        r["이름"] = FEATURE_KO.get(f, f)
        r["종류"] = "포드맵" if f in SUBGROUPS else "포드맵 외 요인"
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_o"] = out["판정"].map(LABEL_ORDER)
    return out.sort_values(["_o", "p_greater"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def ingredient_table(df: pd.DataFrame, min_n: int = 5, seed: int = 0) -> pd.DataFrame:
    """재료 단위 분석(보조). 한국 음식은 마늘·양파·대파가 함께 들어가므로 해석에 주의."""
    rng = np.random.default_rng(seed)
    sets = df["ingredients"].str.split("|")
    counts = sets.explode().value_counts()
    rows = []
    for name, n in counts.items():
        if not name or n < min_n:
            continue
        x = sets.apply(lambda s: name in s).values
        r = bayes_compare(x, df["outcome"].values, rng)
        r["재료"] = name
        r["판정"] = classify(r, min_n)
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("p_greater", ascending=False).reset_index(drop=True)


def fit_logistic_l2(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2 벌점 로지스틱 회귀 (뉴턴-랩슨). scikit-learn LogisticRegression(C=1/lam)과 같은 목적함수.
    절편은 벌점 없이 추정하고, 반환값은 절편을 뺀 계수."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    pen = np.full(Xb.shape[1], lam)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) + pen * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w[1:]


def adjusted_model(df: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame | None:
    """교란요인(스트레스·수면·월경)을 함께 넣은 벌점 로지스틱 회귀. 오즈비와 부트스트랩 95% CI."""
    cols = [f"exp_{f}" for f in SUBGROUPS + FLAGS] + list(CONFOUNDERS)
    cols = [c for c in cols if c in df and df[c].nunique() > 1]
    y = df["outcome"].values
    if len(cols) == 0 or len(np.unique(y)) < 2 or len(df) < 20:
        return None
    X = df[cols].astype(float).values
    coefs = [fit_logistic_l2(X, y)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(fit_logistic_l2(X[idx], y[idx]))
    boots = np.array(boots)
    names = {**{f"exp_{k}": v for k, v in FEATURE_KO.items()}, **CONFOUNDERS}
    out = pd.DataFrame({
        "요인": [names.get(c, c) for c in cols],
        "오즈비(OR)": np.exp(coefs[0]),
        "95% 하한": np.exp(np.quantile(boots, 0.025, axis=0)),
        "95% 상한": np.exp(np.quantile(boots, 0.975, axis=0)),
    })
    return out.sort_values("오즈비(OR)", ascending=False).reset_index(drop=True)


def naive_top(df: pd.DataFrame, features: list[str] | None = None) -> str | None:
    """비교 대상(단순 방법): '증상이 있었던 식사에 가장 자주 들어 있던 성분'을 1위로 지목."""
    features = features or (SUBGROUPS + FLAGS)
    sym = df[df["outcome"] == 1]
    if sym.empty:
        return None
    counts = {f: int(sym[f"exp_{f}"].sum()) for f in features}
    return max(counts, key=counts.get)
