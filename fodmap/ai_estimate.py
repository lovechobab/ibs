"""(선택 기능) AI로 메뉴의 1인분 재료 추정.

- 레시피 DB에 없는 메뉴일 때만 사용하는 보조 수단.
- 서버 비밀설정(Streamlit Secrets)에 ANTHROPIC_API_KEY가 있을 때만 화면에 나타난다.
  키가 없으면 이 기능만 숨겨지고 나머지 앱은 그대로 동작한다 → 심사위원은 키가 필요 없다.
- AI에는 '메뉴 이름'만 보낸다. 증상·식사 기록 같은 개인 건강정보는 절대 보내지 않는다.
- 결과는 '추정'으로 표시하고, 사용자가 표에서 확인·수정한 뒤에만 식사에 반영된다.
- AI가 돌려준 재료명도 parser.Matcher로 다시 DB에 매칭한다 (AI가 포드맵을 직접 판정하지 않음).
"""
from __future__ import annotations

import json
import re

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5"   # 사용 가능한 모델명은 Anthropic 문서에서 확인 후 Secrets의 ANTHROPIC_MODEL로 변경

PROMPT = """당신은 한국 가정식·외식 레시피에 익숙한 영양사 보조입니다.
메뉴 "{menu}"의 일반적인 **1인분** 재료와 그램(g) 양을 추정하세요.

규칙:
- 양념(마늘, 양파, 파, 고추장, 간장, 설탕 등)과 숨은 재료(밀가루 반죽, 소스, 육수 재료)까지 빠짐없이 포함
- 가능하면 아래 목록의 재료명을 그대로 사용하고, 없으면 일반적인 한국어 재료명 사용
- 물, 소금, 후추처럼 의미 없는 재료는 생략 가능
- 메뉴를 모르겠으면 ingredients를 빈 배열로
- 반드시 아래 JSON 형식만 출력 (설명 문장 금지)

{{"dish": "메뉴명", "ingredients": [{{"name": "재료명", "grams": 숫자}}], "note": "가정한 레시피 한 줄 설명"}}

재료 목록: {names}"""


def estimate(menu: str, api_key: str, db_names: list[str], model: str | None = None, timeout: float = 30.0) -> dict:
    import requests

    menu = menu.strip()[:40]  # 메뉴 이름만, 길이 제한
    body = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": 800,
        "messages": [{"role": "user", "content": PROMPT.format(menu=menu, names=", ".join(db_names))}],
    }
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    res = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
    res.raise_for_status()
    text = "".join(b.get("text", "") for b in res.json().get("content", []) if b.get("type") == "text")
    return parse_response(text)


def parse_response(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("AI 응답에서 JSON을 찾지 못했어요.")
    data = json.loads(m.group(0))
    items = []
    for it in data.get("ingredients", []):
        try:
            g = float(it.get("grams", 0))
        except (TypeError, ValueError):
            continue
        name = str(it.get("name", "")).strip()
        if name and 0 < g <= 2000:
            items.append({"name": name, "grams": g})
    return {"dish": data.get("dish", ""), "ingredients": items, "note": str(data.get("note", ""))[:200]}


def to_recipe_text(result: dict) -> str:
    """AI 결과를 파서가 읽는 '재료 1g' 텍스트로 바꿔 같은 매칭 경로를 타게 한다."""
    return "\n".join(f"{it['name']} {it['grams']:g}g" for it in result["ingredients"])
