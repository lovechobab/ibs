"""포드맵 식품 DB 로딩과 식사 판정 로직.

판정 규칙 (보고서에 그대로 설명할 수 있도록 단순하게 설계)
-------------------------------------------------------------
1) 재료별 기준
   - green_max_g 이하  → 🟢 저포드맵
   - red_min_g 이상    → 🔴 고포드맵
   - 그 사이           → 🟡 주의
2) 한 끼 누적(stacking)
   - 재료 i의 '부하(load)' = 먹은 양 / red_min_g  (고포드맵 기준량의 몇 배인가)
   - 같은 하위그룹(예: 프럭탄)끼리 부하를 더한다.
   - 하위그룹 부하 ≥ 1.0 → 🔴
   - '주의' 재료가 있거나, 2개 이상 재료가 쌓여 부하 > 0.5 → 🟡, 나머지 🟢
     (재료 1개만 있을 때는 그 재료 자체의 기준을 따른다)
   - 한 끼 전체 판정 = 가장 나쁜 하위그룹의 판정
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SUBGROUPS = ["fructan", "GOS", "lactose", "fructose", "sorbitol", "mannitol"]
SUBGROUP_KO = {
    "fructan": "프럭탄",
    "GOS": "갈락토올리고당(GOS)",
    "lactose": "유당",
    "fructose": "과당(과잉 과당)",
    "sorbitol": "소르비톨",
    "mannitol": "만니톨",
}
FLAGS = ["caps", "fat", "caffeine", "alcohol"]
FLAG_KO = {
    "caps": "매운맛(캡사이신)",
    "fat": "기름진 음식",
    "caffeine": "카페인",
    "alcohol": "알코올",
}
FEATURE_KO = {**SUBGROUP_KO, **FLAG_KO}

STATUS_KO = {"low": "🟢 저포드맵", "moderate": "🟡 주의", "high": "🔴 고포드맵"}
STATUS_RANK = {"low": 0, "moderate": 1, "high": 2}
RANK_STATUS = {v: k for k, v in STATUS_RANK.items()}

HIGH_LOAD = 1.0       # 하위그룹 부하가 이 값 이상이면 고포드맵
EXPOSE_LOAD = 0.5     # 이 값 이상이면 '주의' 및 분석상 '노출'로 간주


def _split(s) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [x.strip() for x in str(s).split(";") if x.strip() and x.strip() != "none"]


def load_ingredients(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "ingredients.csv")
    for col in ["green_max_g", "red_min_g", "serving_g"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["subgroups", "flags", "swap", "confidence"]:
        df[col] = df[col].fillna("").astype(str)
    return df.set_index("name", drop=False)


def load_recipes(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "recipes.csv")
    df["grams"] = pd.to_numeric(df["grams"], errors="coerce")
    return df


def build_lookup(ing: pd.DataFrame) -> dict:
    """빠른 계산을 위해 DataFrame을 dict로 변환."""
    out = {}
    for name, r in ing.iterrows():
        out[name] = {
            "subgroups": _split(r["subgroups"]),
            "green": r["green_max_g"],
            "red": r["red_min_g"],
            "flags": _split(r["flags"]),
            "swap": r["swap"],
        }
    return out


def item_status(info: dict, grams: float) -> str:
    if not info["subgroups"] or pd.isna(info["red"]) or grams <= 0:
        return "low"
    if grams >= info["red"]:
        return "high"
    if grams > info["green"]:
        return "moderate"
    return "low"


def item_load(info: dict, grams: float) -> float:
    if not info["subgroups"] or pd.isna(info["red"]) or info["red"] <= 0:
        return 0.0
    return max(grams, 0.0) / info["red"]


def expand_dish(recipes: pd.DataFrame, dish: str, servings: float = 1.0) -> list[tuple[str, float]]:
    sub = recipes[recipes["dish"] == dish]
    return [(r.ingredient, float(r.grams) * servings) for r in sub.itertuples()]


def assess_meal(items, lookup: dict) -> dict:
    """items: [(재료명, g), ...] → 판정 결과 dict."""
    loads = {s: 0.0 for s in SUBGROUPS}
    worst_item = {s: 0 for s in SUBGROUPS}
    n_contrib = {s: 0 for s in SUBGROUPS}
    flags: set[str] = set()
    rows = []
    for name, grams in items:
        if name not in lookup or grams is None or pd.isna(grams):
            continue
        grams = float(grams)
        info = lookup[name]
        st_ = item_status(info, grams)
        load = item_load(info, grams)
        for s in info["subgroups"]:
            loads[s] += load
            worst_item[s] = max(worst_item[s], STATUS_RANK[st_])
            if load > 0:
                n_contrib[s] += 1
        if grams > 0:
            flags.update(info["flags"])
        rows.append({
            "재료": name,
            "양(g)": round(grams, 1),
            "판정": STATUS_KO[st_],
            "관련 성분": ", ".join(SUBGROUP_KO[s] for s in info["subgroups"]) or "-",
            "부하": round(load, 2),
            "대체 제안": info["swap"] if st_ != "low" else "",
            "_status": st_,
        })
    sg_status = {}
    for s in SUBGROUPS:
        if loads[s] >= HIGH_LOAD:
            sg_status[s] = "high"
        elif worst_item[s] >= 1 or (n_contrib[s] >= 2 and loads[s] > EXPOSE_LOAD):
            sg_status[s] = "moderate"
        else:
            sg_status[s] = "low"
    overall = RANK_STATUS[max([STATUS_RANK[v] for v in sg_status.values()] + [0])]
    return {
        "items": pd.DataFrame(rows),
        "loads": loads,
        "sg_status": sg_status,
        "overall": overall,
        "flags": sorted(flags),
    }
