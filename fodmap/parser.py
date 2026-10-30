"""레시피 재료 문장 → (DB 재료명, g) 변환기.

예)  "다진 마늘 1큰술"      → 마늘 10g
     "연두부 75g(3/4모)"    → 순두부 75g   (g가 적혀 있으면 그 값을 우선)
     "양파 1/2개"           → 양파 100g
     "배추김치 200g"        → 배추 180g + 마늘 6g + 고춧가루 8g + 대파(흰 부분) 6g (복합 재료 분해)
     "소금 약간"            → 계산 제외(포드맵 없음)

매칭 순서: ① 동의어 사전 정확 일치 → ② DB 이름·동의어 부분 일치(가장 긴 것) → ③ 유사도(difflib) 추정 → ④ 실패(확인 필요)
판단이 애매한 파 종류는 '흰 부분'(고포드맵)으로 보수적으로 매칭한다.
"""
from __future__ import annotations

import difflib
import re

IGNORE = "(제외)"

# 원문 표현 → DB 재료명  (띄어쓰기 제거 후 비교)
SYNONYMS = {
    "마늘": ["마늘", "다진마늘", "통마늘", "편마늘", "간마늘", "마늘즙"],
    "마늘기름": ["마늘기름", "마늘오일"],
    "양파": ["양파", "다진양파", "적양파", "양파즙"],
    "대파(흰 부분)": ["대파", "파", "쪽파", "실파", "다진파", "송송썬파", "대파흰부분", "파흰부분", "다진대파"],
    "대파(초록 부분)": ["대파초록부분", "파잎", "대파잎", "파초록부분", "쪽파초록부분"],
    "부추": ["부추", "영양부추"],
    "배추": ["배추", "알배추", "알배기배추", "배춧잎"],
    "양배추": ["양배추"],
    "상추": ["상추", "양상추", "로메인"],
    "무": ["무", "단무지", "무말랭이"],
    "당근": ["당근"],
    "감자": ["감자"],
    "고구마": ["고구마"],
    "애호박": ["애호박", "호박", "주키니"],
    "단호박": ["단호박"],
    "가지": ["가지"],
    "양송이버섯": ["양송이", "양송이버섯"],
    "느타리버섯": ["느타리", "느타리버섯", "새송이", "새송이버섯"],
    "표고버섯": ["표고", "표고버섯", "건표고"],
    "팽이버섯": ["팽이", "팽이버섯"],
    "콩나물": ["콩나물"],
    "숙주": ["숙주", "숙주나물"],
    "시금치": ["시금치"],
    "오이": ["오이"],
    "토마토": ["토마토", "방울토마토", "토마토소스"],
    "브로콜리": ["브로콜리"],
    "콜리플라워": ["콜리플라워"],
    "셀러리": ["셀러리", "샐러리"],
    "파프리카(빨강)": ["파프리카", "피망", "빨강파프리카", "노랑파프리카"],
    "풋고추": ["풋고추", "꽈리고추", "오이고추", "고추"],
    "청양고추": ["청양고추", "홍고추", "베트남고추", "건고추"],
    "고춧가루": ["고춧가루", "고추가루"],
    "깻잎": ["깻잎"],
    "김": ["김", "김밥김", "김가루", "조미김"],
    "미역": ["미역", "건미역", "다시마"],
    "옥수수": ["옥수수", "옥수수콘", "스위트콘"],
    "완두콩": ["완두콩", "완두"],
    "사과": ["사과"], "배": ["배", "배즙"], "수박": ["수박"], "복숭아": ["복숭아"], "망고": ["망고"],
    "바나나(잘 익은 것)": ["바나나"], "딸기": ["딸기"], "귤": ["귤", "오렌지"], "포도": ["포도"],
    "키위": ["키위", "참다래"], "블루베리": ["블루베리"], "체리": ["체리"], "건포도": ["건포도"],
    "흰쌀밥": ["밥", "쌀밥", "흰밥", "흰쌀밥", "공기밥"],
    "현미밥": ["현미밥", "잡곡밥"],
    "떡(쌀떡)": ["떡", "가래떡", "떡볶이떡", "쌀떡", "떡국떡"],
    "밀가루": ["밀가루", "부침가루", "튀김가루", "중력분", "박력분", "강력분", "빵가루"],
    "식빵(밀)": ["식빵", "빵", "모닝빵", "바게트"],
    "밀국수(삶은 것)": ["국수", "소면", "중면", "칼국수면", "우동면", "우동", "스파게티", "파스타", "중화면"],
    "라면사리(유탕면)": ["라면", "라면사리"],
    "쌀국수": ["쌀국수", "쌀면"],
    "당면": ["당면"],
    "메밀면": ["메밀면", "냉면", "냉면사리", "메밀국수"],
    "오트밀": ["오트밀", "귀리"],
    "보리": ["보리", "보리밥"],
    "우유": ["우유", "흰우유"],
    "생크림": ["생크림", "휘핑크림"],
    "요거트(일반)": ["요거트", "요구르트", "플레인요거트"],
    "체다치즈": ["체다치즈", "슬라이스치즈", "치즈"],
    "모짜렐라치즈": ["모짜렐라", "모짜렐라치즈", "피자치즈"],
    "크림치즈": ["크림치즈"],
    "버터": ["버터", "무염버터"],
    "소고기": ["소고기", "쇠고기", "불고기용소고기", "다진소고기", "우둔", "양지", "사태", "차돌박이", "갈비", "소갈비"],
    "돼지고기(앞다리)": ["돼지고기", "앞다리살", "목살", "뒷다리살", "다진돼지고기", "돼지등갈비", "돼지갈비"],
    "삼겹살": ["삼겹살", "오겹살"],
    "닭고기": ["닭", "닭고기", "닭가슴살", "닭다리살", "닭다리", "닭안심", "닭날개"],
    "생선": ["생선", "고등어", "연어", "동태", "대구", "갈치", "삼치", "참치", "명태", "가자미", "조기", "흰살생선"],
    "새우": ["새우", "칵테일새우", "건새우", "대하"],
    "계란": ["계란", "달걀", "달걀노른자", "달걀흰자", "계란물", "메추리알"],
    "두부(단단한 것)": ["두부", "부침두부", "찌개두부"],
    "순두부": ["순두부", "연두부"],
    "삶은 콩(대두)": ["콩", "대두", "서리태", "검은콩"],
    "병아리콩(통조림)": ["병아리콩"],
    "베이크드빈": ["베이크드빈", "베이크드빈스", "베이크드빈즈"],
    "팥": ["팥"],
    "어묵": ["어묵", "오뎅"],
    "햄·소시지": ["햄", "소시지", "스팸", "베이컨", "비엔나소시지"],
    "간장": ["간장", "진간장", "국간장", "양조간장", "조림간장"],
    "된장": ["된장", "쌈장", "춘장", "미소"],
    "고추장": ["고추장"],
    "참기름": ["참기름", "들기름"],
    "식용유": ["식용유", "카놀라유", "올리브유", "포도씨유", "콩기름", "튀김기름", "현미유"],
    "설탕": ["설탕", "흑설탕", "황설탕", "백설탕"],
    "꿀": ["꿀"],
    "액상과당·물엿": ["물엿", "올리고당", "조청", "요리당", "액상과당"],
    "케첩": ["케첩", "케찹"],
    "마요네즈": ["마요네즈"],
    "아메리카노": ["커피", "에스프레소", "아메리카노"],
    "녹차": ["녹차"], "사과주스": ["사과주스"], "맥주": ["맥주"], "소주": ["소주"],
    "땅콩": ["땅콩", "땅콩버터"], "아몬드": ["아몬드"], "캐슈넛": ["캐슈넛", "캐슈너트"], "호두": ["호두"],
    "피스타치오": ["피스타치오"],
    IGNORE: ["물", "육수", "멸치육수", "다시마육수", "사골육수", "소금", "꽃소금", "굵은소금", "후추", "후춧가루",
             "깨", "통깨", "참깨", "깨소금", "생강", "다진생강", "맛술", "미림", "청주", "소주(요리용)", "식초",
             "굴소스", "까나리액젓", "멸치액젓", "액젓", "새우젓", "얼음", "레몬즙", "파슬리", "월계수잎", "계피",
             "전분", "감자전분", "녹말", "옥수수전분", "녹말물", "물녹말"],
}

# 복합 재료 → 구성 재료 비율 (팀 추정치, 1g당)
COMPOSITES = {
    "김치": {"배추": 0.90, "마늘": 0.03, "고춧가루": 0.04, "대파(흰 부분)": 0.03},
    "배추김치": {"배추": 0.90, "마늘": 0.03, "고춧가루": 0.04, "대파(흰 부분)": 0.03},
    "신김치": {"배추": 0.90, "마늘": 0.03, "고춧가루": 0.04, "대파(흰 부분)": 0.03},
    "깍두기": {"무": 0.90, "마늘": 0.03, "고춧가루": 0.04, "대파(흰 부분)": 0.03},
    "양념장": {"간장": 0.6, "마늘": 0.1, "설탕": 0.1, "참기름": 0.1, "대파(흰 부분)": 0.1},
}

# 1큰술(15mL)당 g  (기본값 15)
TBSP_G = {"마늘": 10, "고추장": 17, "된장": 17, "간장": 18, "설탕": 12, "고춧가루": 6, "꿀": 20, "참기름": 13,
          "식용유": 13, "밀가루": 8, "버터": 13, "액상과당·물엿": 20, "케첩": 17, "마요네즈": 14, "대파(흰 부분)": 7,
          "양파": 10, "땅콩": 9}
# 개수 단위 1개당 g
PIECE_G = {"마늘": 5, "양파": 200, "대파(흰 부분)": 100, "대파(초록 부분)": 10, "감자": 150, "고구마": 200,
           "당근": 150, "애호박": 300, "오이": 200, "계란": 50, "두부(단단한 것)": 300, "순두부": 350,
           "청양고추": 10, "풋고추": 15, "표고버섯": 20, "양송이버섯": 15, "느타리버섯": 15, "사과": 250, "배": 500,
           "토마토": 150, "바나나(잘 익은 것)": 120, "흰쌀밥": 210, "현미밥": 210, "식빵(밀)": 26, "깻잎": 2,
           "김": 3, "상추": 10, "라면사리(유탕면)": 110, "가지": 150, "파프리카(빨강)": 150, "새우": 15,
           "어묵": 30, "햄·소시지": 20, "배추": 30, "양배추": 30, "단호박": 800, "귤": 90, "키위": 75,
           "닭고기": 150, "생선": 150, "떡(쌀떡)": 15, "밀국수(삶은 것)": 250, "메밀면": 200, "쌀국수": 200}
SMALL_G = 2.0       # '약간', '조금', '적당량'
HANDFUL_G = 30.0    # '한 줌'

NUM = r"(?:\d+(?:\.\d+)?(?:\s*/\s*\d+)?|[½¼¾⅓⅔]|반|한|두|세|네|다섯)"
UNIT = (r"(?:kg|g|그램|mL|ml|ML|L|l|cc|큰\s*술|큰\s*스푼|숟가락|T|Ts|tbsp|작은\s*술|작은\s*스푼|티스푼|t|tsp|컵|"
        r"개|쪽|톨|알|모|대|뿌리|줄기|장|공기|봉지|봉|마리|포기|잎|줌|조각|토막|인분|팩|캔|송이)")
QTY_RE = re.compile(rf"({NUM})\s*({UNIT})")
GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|그램|mL|ml|ML|cc|L)\b|(\d+(?:\.\d+)?)\s*(kg|g|그램|mL|ml|ML|cc|L)(?=[^a-zA-Z]|$)")
VAGUE_RE = re.compile(r"약간|조금|적당량|적당히|소량|기호에\s*따라|취향껏|한\s*꼬집")
WORD_NUM = {"반": 0.5, "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}


def _norm(s: str) -> str:
    return re.sub(r"[\s\-·•●○◎■□▶※*()\[\]{}<>]", "", s)


def _to_float(q: str) -> float:
    q = q.replace(" ", "")
    if q in WORD_NUM:
        return WORD_NUM[q]
    if "/" in q:
        a, b = q.split("/")
        return float(a) / float(b) if float(b) else 0.0
    return float(q)


class Matcher:
    def __init__(self, db_names: list[str]):
        self.db_names = list(db_names)
        self.alias: dict[str, str] = {}
        for name in self.db_names:
            self.alias[_norm(name)] = name
        for target, words in SYNONYMS.items():
            if target != IGNORE and target not in self.db_names:
                continue
            for w in words:
                self.alias[_norm(w)] = target
        for c in COMPOSITES:
            self.alias[_norm(c)] = f"composite:{c}"
        self._keys = sorted(self.alias, key=len, reverse=True)

    def match(self, raw_name: str) -> tuple[str | None, str]:
        """→ (DB 재료명 또는 IGNORE 또는 'composite:이름' 또는 None, 매칭 방식)"""
        key = _norm(raw_name)
        if not key:
            return None, "실패"
        if key in self.alias:
            return self.alias[key], "자동"
        for k in self._keys:  # 긴 표현부터 부분 일치 ('다진대파' 안의 '대파')
            if len(k) >= 1 and k in key and (len(k) >= 2 or len(key) <= 2):
                return self.alias[k], "자동"
        close = difflib.get_close_matches(key, self._keys, n=1, cutoff=0.75)
        if close:
            return self.alias[close[0]], "추정"
        return None, "실패"


def _grams_for(target: str, line: str, db_serving: dict) -> tuple[float | None, str]:
    """수량 표현을 g으로. g가 직접 적혀 있으면 최우선."""
    m = GRAM_RE.search(line)
    if m:
        val = float(m.group(1) or m.group(3))
        unit = (m.group(2) or m.group(4)).lower()
        return (val * 1000 if unit in ("kg", "l") else val), "직접"
    m = QTY_RE.search(line)
    if m:
        n = _to_float(m.group(1))
        unit = re.sub(r"\s", "", m.group(2))
        if unit in ("큰술", "큰스푼", "숟가락", "T", "Ts", "tbsp"):
            return n * TBSP_G.get(target, 15), "환산"
        if unit in ("작은술", "작은스푼", "티스푼", "t", "tsp"):
            return n * TBSP_G.get(target, 15) / 3, "환산"
        if unit == "컵":
            return n * 200, "환산"
        if unit == "줌":
            return n * HANDFUL_G, "환산"
        if unit == "인분":
            return n * (db_serving.get(target) or 100), "환산"
        piece = PIECE_G.get(target)
        if piece:
            return n * piece, "환산"
        return n * (db_serving.get(target) or 50), "추정"
    if VAGUE_RE.search(line):
        return SMALL_G, "환산"
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", line)
    if m:
        return float(m.group(1)), "추정"
    return None, "없음"


def split_items(text: str) -> list[str]:
    text = text.replace("\r", "\n")
    parts = re.split(r"[\n,，;]|(?<=[a-zA-Z가-힣)\]])\s*·\s*", text)
    out = []
    for p in parts:
        p = p.strip(" -•●○◎■□▶※*\t")
        if not p:
            continue
        p = re.sub(r"^[\[\(<]?[^\]\)>:]{1,10}[\]\)>]\s*", "", p) if re.match(r"^[\[\(<]", p) else p
        if ":" in p:  # '양념 : 간장 1큰술' → '간장 1큰술'
            head, tail = p.split(":", 1)
            p = tail.strip() if tail.strip() else head.strip()
        if p:
            out.append(p)
    return out


def _name_part(line: str) -> str:
    name = re.split(rf"\s*(?={NUM}\s*{UNIT})|\s*\d|약간|조금|적당량|적당히|소량|\(", line, maxsplit=1)[0]
    return name.strip()


def parse_recipe_text(text: str, db_names: list[str], db_serving: dict | None = None,
                      dish_name: str | None = None) -> list[dict]:
    """반환: [{원문, 재료, 양(g), 상태}]  상태 ∈ 자동 / 추정 / 확인 필요 / 제외"""
    db_serving = db_serving or {}
    matcher = Matcher(db_names)
    rows = []
    for line in split_items(text):
        name = _name_part(line)
        if not name or (dish_name and _norm(name) == _norm(dish_name) and not re.search(r"\d", line)):
            continue
        has_qty = bool(QTY_RE.search(line) or GRAM_RE.search(line) or VAGUE_RE.search(line))
        target, how = matcher.match(name)
        if target is None:
            if not has_qty and len(_norm(name)) <= 6:   # '고명', '양념장 재료' 같은 소제목
                continue
            g, _ = _grams_for("", line, db_serving)
            rows.append({"원문": line, "재료": None, "양(g)": g, "상태": "확인 필요"})
            continue
        if target == IGNORE:
            rows.append({"원문": line, "재료": IGNORE, "양(g)": 0.0, "상태": "제외"})
            continue
        if target.startswith("composite:"):
            comp = target.split(":", 1)[1]
            g, gh = _grams_for("", line, db_serving)
            g = g if g is not None else 100.0
            for sub, frac in COMPOSITES[comp].items():
                rows.append({"원문": f"{line} → {comp} 분해", "재료": sub, "양(g)": round(g * frac, 1),
                             "상태": "추정"})
            continue
        g, gh = _grams_for(target, line, db_serving)
        status = "자동" if how == "자동" and gh in ("직접", "환산") else "추정"
        if g is None:
            g, status = float(db_serving.get(target) or 50), "확인 필요"
        rows.append({"원문": line, "재료": target, "양(g)": round(g, 1), "상태": status})
    return rows


def summarize(rows: list[dict]) -> dict:
    c = {"자동": 0, "추정": 0, "확인 필요": 0, "제외": 0}
    for r in rows:
        c[r["상태"]] = c.get(r["상태"], 0) + 1
    return c
