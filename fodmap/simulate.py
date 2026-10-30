"""가상 환자(합성 데이터) 생성기.

실제 환자 데이터를 쓰지 않고, '정답 트리거'를 미리 정해 둔 가상 환자의 식사·증상 일지를 만든다.
분석 알고리즘이 정답을 얼마나 찾아내는지 검증하는 데 사용한다 (validation/run_validation.py).

증상 발생 모델 (가정)
  logit P(식후 증상) = b0 + Σ β·min(하위그룹 부하, 2)   (정답 트리거 하위그룹)
                         + Σ β·[포드맵 외 요인 포함]     (정답 트리거 요인)
                         + 0.7·[스트레스 높음] + 0.5·[월경 기간]
  증상은 식후 2~7시간 사이에 나타난다고 가정. 그 외 하루 6% 확률로 무관한 배경 증상.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .db import FLAGS, SUBGROUPS, assess_meal, build_lookup, expand_dish, load_ingredients, load_recipes

BREAKFAST = ["토스트", "시리얼+우유", "요거트+과일", "계란말이", "과일(사과 1개)", "카페라떼", None, None]
MAIN = ["김치찌개", "된장찌개", "제육볶음", "불고기", "비빔밥", "김밥(1줄)", "라면", "떡볶이", "삼겹살 구이",
        "미역국", "콩나물국", "잡채", "순두부찌개", "닭갈비", "물냉면", "칼국수", "김치볶음밥", "짜장면",
        "후라이드 치킨", "피자(2조각)", "닭가슴살 샐러드", "쌀국수(저포드맵 예시)"]
RICE_SIDE = {"김치찌개", "된장찌개", "제육볶음", "불고기", "계란말이", "미역국", "콩나물국", "잡채",
             "순두부찌개", "닭갈비", "삼겹살 구이"}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate_patient(days: int = 56, triggers: dict | None = None, seed: int = 0, female: bool = True,
                     start: date | None = None, ing=None, recipes=None, lookup=None,
                     b0: float = -3.2) -> dict:
    triggers = triggers or {}
    rng = np.random.default_rng(seed)
    ing = ing if ing is not None else load_ingredients()
    recipes = recipes if recipes is not None else load_recipes()
    lookup = lookup or build_lookup(ing)
    start = start or date(2026, 10, 1)
    cycle_offset = int(rng.integers(0, 28))

    meal_rows, sym_rows, daily_rows = [], [], []
    for d in range(days):
        day = start + timedelta(days=d)
        day0 = datetime(day.year, day.month, day.day)
        stress = int(np.clip(round(rng.normal(4, 2)), 0, 10))
        sleep = float(np.clip(round(rng.normal(6.8, 1.0), 1), 3, 10))
        mens = bool(female and ((d + cycle_offset) % 28) < 5)
        daily_rows.append({"date": day.isoformat(), "stress": stress, "sleep_h": sleep, "menstruation": mens})

        for label, hour, menu in [("아침", 8.0, BREAKFAST), ("점심", 12.5, MAIN), ("저녁", 19.0, MAIN)]:
            dish = menu[rng.integers(0, len(menu))]
            if dish is None:
                continue
            t = day0 + timedelta(hours=float(hour + rng.normal(0, 0.4)))
            t = t.replace(second=0, microsecond=0)
            items = expand_dish(recipes, dish)
            if dish in RICE_SIDE:
                items.append(("흰쌀밥", 210.0))
            if label == "점심" and rng.random() < 0.5:
                items.append(("아메리카노", 350.0))
            if label == "저녁" and rng.random() < 0.15:
                items.append(("맥주", 500.0) if rng.random() < 0.5 else ("소주", 180.0))
            mid = f"sim{seed}_{d:03d}_{label}"
            for name, g in items:
                meal_rows.append({"meal_id": mid, "datetime": t.isoformat(), "meal_label": label,
                                  "dish": dish, "ingredient": name, "grams": float(g)})

            a = assess_meal(items, lookup)
            logit = b0 + 0.7 * (stress >= 7) + 0.5 * mens
            for f, beta in triggers.items():
                if f in SUBGROUPS:
                    logit += beta * min(a["loads"][f], 2.0)
                elif f in FLAGS and f in a["flags"]:
                    logit += beta
            if rng.random() < _sigmoid(logit):
                st_ = t + timedelta(hours=float(rng.uniform(2, 7)))
                sym_rows.append({
                    "datetime": st_.replace(second=0, microsecond=0).isoformat(),
                    "pain": int(rng.integers(4, 9)), "bloating": int(rng.integers(3, 9)),
                    "diarrhea": int(rng.integers(0, 7)), "constipation": 0, "note": "",
                })
        if rng.random() < 0.06:  # 음식과 무관한 배경 증상
            st_ = day0 + timedelta(hours=float(rng.uniform(9, 22)))
            sym_rows.append({"datetime": st_.replace(second=0, microsecond=0).isoformat(),
                             "pain": int(rng.integers(3, 7)), "bloating": int(rng.integers(2, 6)),
                             "diarrhea": 0, "constipation": int(rng.integers(0, 4)), "note": ""})

    meal_items = pd.DataFrame(meal_rows)
    meal_items["datetime"] = pd.to_datetime(meal_items["datetime"])
    symptoms = pd.DataFrame(sym_rows, columns=["datetime", "pain", "bloating", "diarrhea", "constipation", "note"])
    symptoms["datetime"] = pd.to_datetime(symptoms["datetime"])
    return {"meal_items": meal_items, "symptoms": symptoms, "daily": pd.DataFrame(daily_rows)}
