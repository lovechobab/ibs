"""기록 저장소: 서버에 저장하지 않고 브라우저 세션(st.session_state)에만 보관.
사용자는 JSON 파일로 내려받아 보관하고, 다음에 다시 불러온다 (개인 건강정보 비저장 원칙)."""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

MEAL_COLS = ["meal_id", "datetime", "meal_label", "dish", "ingredient", "grams"]
SYM_COLS = ["datetime", "pain", "bloating", "diarrhea", "constipation", "note"]
DAILY_COLS = ["date", "stress", "sleep_h", "menstruation"]


def _empty(cols):
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def init_state():
    ss = st.session_state
    ss.setdefault("meal_items", _empty(MEAL_COLS))
    ss.setdefault("symptoms", _empty(SYM_COLS))
    ss.setdefault("daily", _empty(DAILY_COLS))
    ss.setdefault("challenges", [])
    ss.setdefault("current_meal", pd.DataFrame({"ingredient": pd.Series(dtype="object"),
                                                "grams": pd.Series(dtype="float")}))
    ss.setdefault("editor_ver", 0)
    ss.setdefault("pending_add", [])
    ss.setdefault("demo_info", None)


def _normalize(ss):
    if not ss.meal_items.empty:
        ss.meal_items["datetime"] = pd.to_datetime(ss.meal_items["datetime"])
        ss.meal_items["grams"] = ss.meal_items["grams"].astype(float)
    if not ss.symptoms.empty:
        ss.symptoms["datetime"] = pd.to_datetime(ss.symptoms["datetime"])


def add_meal(dt: datetime, label: str, items: pd.DataFrame, dish: str = ""):
    ss = st.session_state
    mid = uuid.uuid4().hex[:10]
    rows = pd.DataFrame({"meal_id": mid, "datetime": pd.Timestamp(dt), "meal_label": label, "dish": dish,
                         "ingredient": items["ingredient"].values, "grams": items["grams"].astype(float).values})
    ss.meal_items = pd.concat([ss.meal_items, rows], ignore_index=True) if not ss.meal_items.empty else rows
    _normalize(ss)
    return mid


def delete_meal(meal_id: str):
    ss = st.session_state
    ss.meal_items = ss.meal_items[ss.meal_items["meal_id"] != meal_id].reset_index(drop=True)


def add_symptom(dt: datetime, pain, bloating, diarrhea, constipation, note=""):
    ss = st.session_state
    row = pd.DataFrame([{"datetime": pd.Timestamp(dt), "pain": pain, "bloating": bloating,
                         "diarrhea": diarrhea, "constipation": constipation, "note": note}])
    ss.symptoms = pd.concat([ss.symptoms, row], ignore_index=True) if not ss.symptoms.empty else row
    _normalize(ss)


def upsert_daily(d: date, stress, sleep_h, menstruation):
    ss = st.session_state
    df = ss.daily
    if not df.empty:
        df = df[df["date"].astype(str) != d.isoformat()]
    row = pd.DataFrame([{"date": d.isoformat(), "stress": stress, "sleep_h": sleep_h, "menstruation": menstruation}])
    ss.daily = pd.concat([df, row], ignore_index=True) if not df.empty else row


def to_json() -> str:
    ss = st.session_state

    def recs(df):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M")
        return df.to_dict("records")

    return json.dumps({
        "app": "속편한 식탁 (14조)", "version": 1, "exported_at": datetime.now().isoformat(timespec="minutes"),
        "meal_items": recs(ss.meal_items), "symptoms": recs(ss.symptoms), "daily": recs(ss.daily),
        "challenges": ss.challenges,
    }, ensure_ascii=False, indent=1)


def from_json(text: str):
    data = json.loads(text)
    ss = st.session_state
    ss.meal_items = pd.DataFrame(data.get("meal_items", []), columns=MEAL_COLS) if data.get("meal_items") else _empty(MEAL_COLS)
    ss.symptoms = pd.DataFrame(data.get("symptoms", []), columns=SYM_COLS) if data.get("symptoms") else _empty(SYM_COLS)
    ss.daily = pd.DataFrame(data.get("daily", []), columns=DAILY_COLS) if data.get("daily") else _empty(DAILY_COLS)
    ss.challenges = data.get("challenges", [])
    ss.demo_info = None
    _normalize(ss)


def reset():
    for k in ["meal_items", "symptoms", "daily", "challenges", "current_meal", "demo_info"]:
        st.session_state.pop(k, None)
    init_state()


DEMO_TRIGGERS = {"lactose": 2.2, "caps": 1.6}


def load_demo(ing, recipes, days: int = 28):
    from .simulate import simulate_patient

    start = date.today() - timedelta(days=days)
    sim = simulate_patient(days=days, triggers=DEMO_TRIGGERS, seed=14, start=start, ing=ing, recipes=recipes)
    ss = st.session_state
    ss.meal_items, ss.symptoms, ss.daily = sim["meal_items"], sim["symptoms"], sim["daily"]
    ss.challenges = []
    ss.demo_info = {"days": days, "triggers": list(DEMO_TRIGGERS)}
    _normalize(ss)
