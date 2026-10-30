"""식품안전나라 '조리식품의 레시피 DB'(COOKRCP01)를 data/public_recipes.csv로 저장.

팀원 1명이 한 번만 실행하면 된다. 저장된 CSV를 GitHub에 올리면 앱은 키 없이 검색한다.

1) https://www.foodsafetykorea.go.kr → 회원가입 → 공공데이터활용 → 'COOKRCP01 조리식품의 레시피 DB' 인증키 신청(무료)
2) 실행:  python scripts/download_public_recipes.py 발급받은인증키
3) 이용 조건(출처 표기 등)을 확인하고 README·보고서에 출처를 적는다.
※ 인증키는 코드나 GitHub에 절대 올리지 말 것.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fodmap.recipe_source import PUBLIC_CSV, download_all  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("사용법: python scripts/download_public_recipes.py <인증키>")
    n = download_all(sys.argv[1])
    print(f"{n}개 레시피 저장 → {PUBLIC_CSV}")
