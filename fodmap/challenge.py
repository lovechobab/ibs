"""재도입(Re-challenge) 단계 안내와 판정.

일반적인 저포드맵 프로토콜(Whelan 2018 등)을 단순화:
  - 제한기(2~6주)로 증상이 안정된 뒤 하위그룹을 하나씩 테스트
  - 1~3일차: 같은 음식을 소량 → 중간 → 평소량으로 늘림
  - 4~6일차: 휴지기(제한식 유지) 후 다음 하위그룹
판정: 그날 최고 증상 점수가 평소(기저) 대비 3점 이상 오르면 '반응'으로 보고 그 날 이후 중단.
※ 실제 진행은 의사·영양사와 상의. 아래 음식·용량은 교육용 예시.
"""
from __future__ import annotations

PLANS = {
    "fructan": {"food": "마늘(또는 밀 식빵)", "doses": ["마늘 1/4쪽(약 1g)", "마늘 1/2쪽(약 2g)", "마늘 1쪽(약 4g)"]},
    "lactose": {"food": "우유", "doses": ["60 mL", "125 mL", "250 mL"]},
    "fructose": {"food": "꿀", "doses": ["1작은술", "2작은술", "1큰술"]},
    "sorbitol": {"food": "복숭아", "doses": ["1/4개", "1/2개", "1개"]},
    "mannitol": {"food": "양송이버섯", "doses": ["2개(약 30g)", "4개(약 60g)", "1컵(약 120g)"]},
    "GOS": {"food": "병아리콩(통조림, 헹군 것)", "doses": ["1/4컵", "1/2컵", "1컵"]},
}

REACTION_DELTA = 3


def evaluate(baseline: float, day_scores: list[float | None]) -> dict:
    """day_scores: 1~3일차 최고 증상 점수 (아직 안 했으면 None)."""
    done = [s for s in day_scores if s is not None]
    for i, s in enumerate(day_scores):
        if s is None:
            break
        if s - baseline >= REACTION_DELTA:
            return {"result": "반응 있음", "day": i + 1,
                    "message": f"{i + 1}일차 용량에서 증상이 평소보다 {s - baseline:.0f}점 증가했어요. "
                               f"테스트를 멈추고 증상이 가라앉을 때까지 제한식을 유지하세요. "
                               f"이보다 적은 양({'없음' if i == 0 else f'{i}일차 용량'})까지는 견딜 가능성이 있어요."}
    if len(done) == 3:
        return {"result": "내성 양호", "day": None,
                "message": "3일 동안 의미 있는 증상 증가가 없었어요. 이 성분은 평소 식단에 다시 넣어도 될 가능성이 높아요."}
    return {"result": "진행 중", "day": None, "message": f"{len(done)}/3일 기록됨."}
