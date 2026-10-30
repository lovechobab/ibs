"""핵심 로직 단위 테스트:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fodmap import analysis, challenge, db  # noqa: E402

ING = db.load_ingredients()
REC = db.load_recipes()
LK = db.build_lookup(ING)


def overall(items):
    return db.assess_meal(items, LK)["overall"]


def test_db_integrity():
    assert ING.index.is_unique
    for name, r in ING.iterrows():
        sg = db._split(r["subgroups"])
        assert all(s in db.SUBGROUPS for s in sg), name
        assert all(f in db.FLAGS for f in db._split(r["flags"])), name
        if sg:
            assert r["red_min_g"] > r["green_max_g"] >= 0, name
    assert set(REC["ingredient"]) <= set(ING.index), set(REC["ingredient"]) - set(ING.index)


def test_known_answers():
    assert overall([("양파", 50)]) == "high"
    assert overall([("마늘", 5)]) == "high"
    assert overall([("당근", 300)]) == "low"
    assert overall([("식빵(밀)", 26)]) == "low"        # 1장
    assert overall([("식빵(밀)", 52)]) == "high"       # 2장
    assert overall([("우유", 200)]) == "high"
    assert overall([("락토프리 우유", 200)]) == "low"


def test_stacking():
    # 각각은 저포드맵 범위지만 같은 성분(프럭탄)이 쌓이면 주의 이상
    assert overall([("식빵(밀)", 26)]) == "low"
    assert overall([("식빵(밀)", 26), ("밀국수(삶은 것)", 74)]) in ("moderate", "high")


def test_dishes():
    assert overall(db.expand_dish(REC, "김치찌개")) == "high"
    assert overall(db.expand_dish(REC, "저포드맵 김치찌개(변형)")) == "low"
    assert overall(db.expand_dish(REC, "쌀국수(저포드맵 예시)")) == "low"


def test_outcome_window():
    feats = pd.DataFrame({"datetime": pd.to_datetime(["2026-10-01 12:00", "2026-10-01 19:00"])})
    sym = pd.DataFrame({"datetime": pd.to_datetime(["2026-10-01 15:00"]), "pain": [6], "bloating": [0],
                        "diarrhea": [0], "constipation": [0]})
    y = analysis.label_outcomes(feats, sym, 1, 8, 4)
    assert y.tolist() == [1, 0]
    sym["pain"] = 2  # 기준 미만
    assert analysis.label_outcomes(feats, sym, 1, 8, 4).tolist() == [0, 0]


def test_bayes_direction():
    rng = np.random.default_rng(0)
    x = np.array([1] * 20 + [0] * 20, bool)
    y = np.array([1] * 15 + [0] * 5 + [1] * 2 + [0] * 18)
    r = analysis.bayes_compare(x, y, rng)
    assert r["p_greater"] > 0.99 and r["rr"] > 3
    assert analysis.classify(r) == "의심 높음"


def test_challenge():
    assert challenge.evaluate(2, [2, 3, 3])["result"] == "내성 양호"
    r = challenge.evaluate(2, [2, 6, None])
    assert r["result"] == "반응 있음" and r["day"] == 2
    assert challenge.evaluate(2, [2, None, None])["result"] == "진행 중"
