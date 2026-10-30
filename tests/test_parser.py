"""레시피 파서·검색·AI 응답 처리 테스트 (네트워크 없이 실행):  pytest -q"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fodmap import ai_estimate, db, parser, recipe_source  # noqa: E402

ING = db.load_ingredients()
NAMES = list(ING.index)
SERV = ING["serving_g"].to_dict()


def parse(text, dish=None):
    return {(r["재료"], r["양(g)"], r["상태"]) for r in parser.parse_recipe_text(text, NAMES, SERV, dish)}


def test_explicit_grams_win():
    rows = parse("연두부 75g(3/4모), 달걀 30g(1/2개)")
    assert ("순두부", 75.0, "자동") in rows and ("계란", 30.0, "자동") in rows


def test_unit_conversion():
    rows = parse("다진 마늘 1큰술\n양파 1/2개\n두부 반 모\n고춧가루 2작은술\n설탕 약간")
    assert ("마늘", 10.0, "자동") in rows
    assert ("양파", 100.0, "자동") in rows
    assert ("두부(단단한 것)", 150.0, "자동") in rows
    assert ("고춧가루", 4.0, "자동") in rows
    assert ("설탕", 2.0, "자동") in rows


def test_conservative_green_onion_and_ignore():
    rows = parse("대파 1대, 물 3컵, 소금 약간")
    assert ("대파(흰 부분)", 100.0, "자동") in rows
    assert (parser.IGNORE, 0.0, "제외") in rows


def test_kimchi_is_decomposed():
    rows = parser.parse_recipe_text("신김치 200g", NAMES, SERV)
    got = {r["재료"]: r["양(g)"] for r in rows}
    assert got["배추"] == 180.0 and got["마늘"] == 6.0


def test_unknown_flagged_not_guessed():
    rows = parser.parse_recipe_text("알 수 없는 소스 2큰술", NAMES, SERV)
    assert rows[0]["재료"] is None and rows[0]["상태"] == "확인 필요"


def test_section_headers_skipped():
    rows = parser.parse_recipe_text("새우두부계란찜\n연두부 75g\n고명\n시금치 10g(3줄기)", NAMES, SERV, "새우두부계란찜")
    assert [r["재료"] for r in rows] == ["순두부", "시금치"]


def test_parsed_meal_is_judged():
    rows = parser.parse_recipe_text("돼지고기 150g, 양파 1/2개, 다진 마늘 1큰술, 고추장 2큰술", NAMES, SERV)
    a = db.assess_meal([(r["재료"], r["양(g)"]) for r in rows], db.build_lookup(ING))
    assert a["overall"] == "high"


def test_builtin_search():
    rec = db.load_recipes()
    assert recipe_source.search_builtin("김치 찌개", rec)[0]["name"] == "김치찌개"
    assert recipe_source.search_builtin("완전히없는메뉴", rec) == []
    assert recipe_source.search_builtin("부대찌개", rec) == []          # '찌개'만 같은 경우 제외
    assert recipe_source.search_builtin("김치볶음밥", rec)[0]["name"] == "김치볶음밥"
    assert recipe_source.search_builtin("짜장", rec)[0]["name"] == "짜장면"


def test_foodsafety_api_shape(monkeypatch):
    sample = {"COOKRCP01": {"total_count": "1", "row": [{
        "RCP_NM": "새우 두부 계란찜", "RCP_PAT2": "반찬", "RCP_WAY2": "찌기", "INFO_WGT": "",
        "RCP_PARTS_DTLS": "새우두부계란찜\n연두부 75g(3/4모), 칵테일새우 20g(5마리), 달걀 30g(1/2개)"}]}}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return sample

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    out = recipe_source.fetch_foodsafety("두부 계란찜", "TESTKEY")
    assert out and out[0]["source"] == recipe_source.PUBLIC_SOURCE and "연두부" in out[0]["text"]


def test_foodsafety_api_failure_is_silent(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", boom)
    assert recipe_source.fetch_foodsafety("김치찌개", "TESTKEY") == []


def test_public_csv_search(tmp_path):
    p = tmp_path / "pub.csv"
    pd.DataFrame({"name": ["부대찌개", "된장국"], "parts": ["햄 50g, 다진마늘 1큰술", "된장 1큰술"]}).to_csv(p, index=False)
    res = recipe_source.search_public("부대 찌개", recipe_source.load_public(p))
    assert res[0]["name"] == "부대찌개"


def test_ai_response_parsing():
    txt = '설명 {"dish": "부대찌개", "ingredients": [{"name": "햄", "grams": 60}, {"name": "다진 마늘", "grams": "8"},' \
          ' {"name": "이상한값", "grams": "많이"}, {"name": "과다", "grams": 99999}], "note": "일반적 레시피"} 끝'
    res = ai_estimate.parse_response(txt)
    assert [i["name"] for i in res["ingredients"]] == ["햄", "다진 마늘"]
    rows = parser.parse_recipe_text(ai_estimate.to_recipe_text(res), NAMES, SERV)
    assert {r["재료"] for r in rows} == {"햄·소시지", "마늘"}
