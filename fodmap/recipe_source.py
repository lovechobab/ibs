"""메뉴 이름 → 레시피 후보 검색.

검색 순서 (위에서 찾으면 아래는 보조)
  1) 앱 내장 요리 (data/recipes.csv)                                   — 항상 사용 가능
  2) 공공 레시피 DB 사본 (data/public_recipes.csv)                       — 팀이 scripts/download_public_recipes.py로 미리 받아 둠
       출처: 식품의약품안전처 식품안전나라 '조리식품의 레시피 DB' (COOKRCP01)
  3) 같은 DB 실시간 API                                                 — 서버 비밀설정에 FOODSAFETY_API_KEY가 있을 때만
  4) (별도) AI 추정 — fodmap/ai_estimate.py

심사위원·사용자는 어떤 키도 입력할 필요가 없다. 키는 팀이 Streamlit Cloud 'Secrets'에만 넣는다.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PUBLIC_CSV = DATA_DIR / "public_recipes.csv"
FOODSAFETY_URL = "http://openapi.foodsafetykorea.go.kr/api/{key}/COOKRCP01/json/{start}/{end}"
PUBLIC_SOURCE = "식품안전나라 조리식품 레시피 DB"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


GENERIC = ["찌개", "전골", "볶음밥", "볶음", "구이", "조림", "무침", "덮밥", "국수", "국", "탕", "밥", "면", "전",
           "찜", "샐러드", "튀김", "죽"]


def _core(s: str) -> str:
    for g in GENERIC:
        if s.endswith(g) and len(s) > len(g):
            return s[: -len(g)]
    return s


def _score(query: str, name: str) -> float:
    """메뉴명 유사도. '찌개'·'볶음' 같은 공통 꼬리말만 같은 경우(부대찌개↔순두부찌개)는 매칭하지 않는다."""
    q, n = _norm(query), _norm(name)
    if not q or not n:
        return 0.0
    if q == n:
        return 1.0
    if len(q) >= 2 and q in n:
        return 0.85 + 0.1 * len(q) / len(n)
    if len(n) >= 2 and n in q:
        return 0.75
    core_q, core_n = _core(q), _core(n)
    if difflib.SequenceMatcher(None, core_q, core_n).ratio() < 0.6:
        return 0.0
    return difflib.SequenceMatcher(None, q, n).ratio()


def search_builtin(menu: str, recipes: pd.DataFrame, limit: int = 3, min_score: float = 0.6) -> list[dict]:
    out = []
    for dish in recipes["dish"].unique():
        s = _score(menu, dish)
        if s >= min_score:
            sub = recipes[recipes["dish"] == dish]
            out.append({"source": "앱 내장 레시피", "name": dish, "score": s, "servings": 1.0,
                        "items": [(r.ingredient, float(r.grams)) for r in sub.itertuples()], "text": None})
    return sorted(out, key=lambda r: -r["score"])[:limit]


def load_public(path: Path = PUBLIC_CSV) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    return df if {"name", "parts"}.issubset(df.columns) else None


def search_public(menu: str, df: pd.DataFrame | None, limit: int = 5, min_score: float = 0.6) -> list[dict]:
    if df is None or df.empty:
        return []
    scores = df["name"].map(lambda n: _score(menu, n))
    top = df.assign(_s=scores).query("_s >= @min_score").sort_values("_s", ascending=False).head(limit)
    return [{"source": PUBLIC_SOURCE, "name": r["name"], "score": float(r["_s"]), "servings": 1.0,
             "items": None, "text": str(r["parts"]), "category": r.get("category", ""),
             "weight": r.get("weight", "")} for _, r in top.iterrows()]


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "name": str(r.get("RCP_NM", "")).strip(),
        "category": r.get("RCP_PAT2", ""),
        "method": r.get("RCP_WAY2", ""),
        "weight": r.get("INFO_WGT", ""),
        "parts": str(r.get("RCP_PARTS_DTLS", "")).strip(),
    } for r in rows])


def fetch_foodsafety(menu: str, key: str, limit: int = 5, timeout: float = 8.0) -> list[dict]:
    """실시간 API 검색. 실패하면 빈 리스트 (앱이 멈추지 않도록)."""
    import requests

    url = FOODSAFETY_URL.format(key=key, start=1, end=limit) + f"/RCP_NM={quote(menu)}"
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        rows = res.json().get("COOKRCP01", {}).get("row", []) or []
    except Exception:  # noqa: BLE001 — 네트워크·형식 오류는 조용히 무시하고 다음 방법으로
        return []
    df = _rows_to_df(rows)
    return search_public(menu, df, limit=limit, min_score=0.0) if not df.empty else []


def download_all(key: str, out: Path = PUBLIC_CSV, batch: int = 1000, max_rows: int = 5000) -> int:
    """공공 레시피 DB 전체를 CSV로 저장 (scripts/download_public_recipes.py에서 사용)."""
    import requests

    rows = []
    for start in range(1, max_rows + 1, batch):
        url = FOODSAFETY_URL.format(key=key, start=start, end=start + batch - 1)
        data = requests.get(url, timeout=30).json().get("COOKRCP01", {})
        got = data.get("row", []) or []
        rows.extend(got)
        total = int(data.get("total_count", 0) or 0)
        if not got or len(rows) >= total:
            break
    df = _rows_to_df(rows).drop_duplicates("name")
    df["source"] = PUBLIC_SOURCE
    df.to_csv(out, index=False)
    return len(df)
