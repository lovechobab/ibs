"""속편한 식탁 — 한국형 저포드맵 식사 판정 · 개인 트리거 탐색기 (14조)
실행: streamlit run app.py
"""
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fodmap import ai_estimate, analysis, challenge, db, parser, recipe_source, storage

st.set_page_config(page_title="속편한 식탁 | 저포드맵 트리거 탐색기", page_icon="🥗", layout="wide")

ROOT = Path(__file__).resolve().parent
DISCLAIMER = ("교육용 프로토타입입니다. 진단·치료 도구가 아니며, 포드맵 함량은 공개 문헌을 바탕으로 한 **근사치**입니다. "
              "식단 변경은 반드시 의사·영양사와 상의하세요.")
RED_FLAGS = [
    "대변에 피가 섞이거나 검은 변을 본 적이 있다",
    "의도하지 않은 체중 감소가 있다",
    "자다가 깰 정도의 복통·설사가 있다",
    "50세 이후 처음 배변 습관이 바뀌었다",
    "빈혈이나 원인 모를 발열이 있다",
    "대장암·염증성 장질환 가족력이 있다",
]
STATUS_COLOR = {"low": "#2e9d5b", "moderate": "#e0a100", "high": "#d64545"}
LABEL_COLOR = {"의심 높음": "#d64545", "가능성 있음": "#e0a100", "근거 약함": "#8a8f98",
               "비교 불가(거의 매번 섭취)": "#5b7db1", "데이터 부족": "#b8bcc4"}


@st.cache_data
def get_data():
    ing, rec = db.load_ingredients(), db.load_recipes()
    return ing, rec


ING, REC = get_data()
LOOKUP = db.build_lookup(ING)
storage.init_state()
ss = st.session_state


def fmt_sg(s):
    return db.FEATURE_KO.get(s, s)


@st.cache_data(show_spinner=False)
def cached_dataset(meal_items: pd.DataFrame, symptoms: pd.DataFrame, daily: pd.DataFrame, lag_min, lag_max, thr):
    return analysis.build_dataset(meal_items, symptoms, daily, ING, lag_min, lag_max, thr, lookup=LOOKUP)


def dataset(lag_min=1, lag_max=8, thr=4):
    return cached_dataset(ss.meal_items, ss.symptoms, ss.daily, lag_min, lag_max, thr)


# ───────────────────────── 사이드바 ─────────────────────────
st.sidebar.title("🥗 속편한 식탁")
PAGES = ["🏠 홈", "🍽️ 식사 판정·기록", "📝 증상·생활 기록", "📊 트리거 분석", "🔁 재도입 챌린지", "🧪 알고리즘 검증", "📚 데이터·출처"]
page = st.sidebar.radio("메뉴", PAGES, label_visibility="collapsed")
st.sidebar.divider()
n_meals = ss.meal_items["meal_id"].nunique() if not ss.meal_items.empty else 0
st.sidebar.metric("기록된 식사", f"{n_meals}끼")
st.sidebar.metric("증상 기록", f"{len(ss.symptoms)}건")
if ss.demo_info:
    st.sidebar.info("지금 **가상 사용자 데모 데이터**를 보고 있어요.")
st.sidebar.download_button("💾 내 기록 저장(JSON)", storage.to_json(), file_name="속편한식탁_기록.json",
                           mime="application/json", use_container_width=True)
st.sidebar.caption("⚠️ " + DISCLAIMER)
st.sidebar.caption("© 2026 14조 · 생물정보학및소프트웨어코딩")


# ───────────────────────── 홈 ─────────────────────────
def page_home():
    st.title("🥗 속편한 식탁")
    st.markdown("#### 과민성 장증후군(IBS)을 위한 **한국형 저포드맵 식사 판정 · 개인 트리거 탐색기**")
    st.warning(DISCLAIMER, icon="⚠️")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("""
**왜 필요한가요?**
마늘·양파·대파·밀가루·우유처럼 한국 식단에 기본으로 들어가는 재료가 대부분 **고포드맵**입니다.
그런데 기존 저포드맵 앱은 서양 음식 위주라 김치찌개, 제육볶음 같은 음식을 판단하기 어렵습니다.
또 사람마다 민감한 성분이 달라서 **"나는 무엇을 피해야 하는지"** 직접 찾아야 합니다.

**사용 순서**
1. **🍽️ 식사 판정·기록**: 먹은 요리·재료와 양을 넣으면 🟢/🟡/🔴로 판정하고 대체 재료를 알려줘요.
2. **📝 증상·생활 기록**: 복통·팽만 등 증상과 스트레스·수면·월경을 기록해요.
3. **📊 트리거 분석**: 2주 이상 쌓이면, 어떤 성분 뒤에 증상이 잦았는지 통계적으로 계산해요.
4. **🔁 재도입 챌린지**: 의심 성분을 하나씩 다시 먹어 보며 확인해요 (표준 저포드맵 3단계).
""")
    with c2:
        with st.container(border=True):
            st.markdown("**🚩 먼저 확인하세요: 경고 증상**")
            checks = [st.checkbox(x, key=f"rf_{i}") for i, x in enumerate(RED_FLAGS)]
            if any(checks):
                st.error("경고 증상이 있다면 이 앱보다 **먼저 소화기내과 진료**를 받으세요. "
                         "IBS가 아닌 다른 질환(염증성 장질환, 대장암, 셀리악병 등)일 수 있습니다.")
            else:
                st.caption("이 앱은 의사에게 IBS 진단을 받은 분을 위한 보조 도구입니다.")

    st.divider()
    st.subheader("시작하기")
    if ss.get("_flash"):
        st.success(ss.pop("_flash"))
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**🧪 데모 데이터로 체험**")
        st.caption("유당·매운맛에 민감하도록 설정한 **가상 사용자**의 28일 기록을 불러옵니다. "
                   "앱이 정답을 찾아내는지 확인해 보세요.")
        if st.button("가상 사용자 데이터 불러오기", type="primary", use_container_width=True):
            storage.load_demo(ING, REC)
            ss["_flash"] = "가상 사용자 데이터를 불러왔어요! 왼쪽 메뉴의 📊 트리거 분석을 눌러 보세요."
            st.rerun()
    with c2, st.container(border=True):
        st.markdown("**📂 저장한 기록 불러오기**")
        up = st.file_uploader("JSON 파일", type="json", label_visibility="collapsed")
        if up is not None and ss.get("_loaded_file") != up.file_id:
            try:
                storage.from_json(up.getvalue().decode("utf-8"))
                ss["_loaded_file"] = up.file_id
                ss["_flash"] = "저장한 기록을 불러왔어요."
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"불러오기 실패: {e}")
    with c3, st.container(border=True):
        st.markdown("**🔒 개인정보**")
        st.caption("기록은 서버에 저장되지 않고 이 브라우저 창에만 머무릅니다. "
                   "창을 닫기 전에 왼쪽의 **내 기록 저장**으로 파일을 받아 두세요.")
        if st.button("모든 기록 지우기", use_container_width=True):
            storage.reset()
            st.rerun()


# ───────────────────────── 식사 판정·기록 ─────────────────────────
def render_assessment(items):
    a = db.assess_meal(items, LOOKUP)
    color = STATUS_COLOR[a["overall"]]
    st.markdown(f"<div style='padding:14px 18px;border-radius:12px;background:{color}1a;border:2px solid {color}'>"
                f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>{db.STATUS_KO[a['overall']]}</span>"
                f"</div>", unsafe_allow_html=True)
    loads = pd.DataFrame({"성분": [fmt_sg(s) for s in db.SUBGROUPS],
                          "부하": [a["loads"][s] for s in db.SUBGROUPS],
                          "판정": [a["sg_status"][s] for s in db.SUBGROUPS]})
    fig = go.Figure(go.Bar(x=loads["부하"].clip(upper=3), y=loads["성분"], orientation="h",
                           marker_color=[STATUS_COLOR[s] for s in loads["판정"]],
                           text=[f"{v:.2f}" for v in loads["부하"]], textposition="outside",
                           hovertemplate="%{y}: 부하 %{text}<extra></extra>"))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#d64545", annotation_text="고포드맵 기준(1.0)",
                  annotation_position="top")
    fig.update_layout(height=260, margin=dict(l=10, r=30, t=30, b=10), xaxis_title="부하 (고포드맵 기준량의 몇 배)",
                      xaxis_range=[0, 3.3], yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("부하 = 먹은 양 ÷ 고포드맵 기준량. 같은 성분은 한 끼 안에서 **누적(stacking)** 됩니다. 3 이상은 3으로 표시.")
    if not a["items"].empty:
        st.dataframe(a["items"].drop(columns="_status"), hide_index=True, use_container_width=True)
        swaps = a["items"][a["items"]["대체 제안"] != ""]
        if not swaps.empty:
            st.info("🔄 **대체 제안**  \n" + "  \n".join(f"- {r['재료']} → {r['대체 제안']}" for _, r in swaps.iterrows()))
    if a["flags"]:
        st.warning("🌶️ **포드맵 외 자극 요인**: " + ", ".join(db.FLAG_KO[f] for f in a["flags"])
                   + " — 포드맵은 아니지만 IBS 증상을 악화시킬 수 있어요.")
    return a


# ───────────────────────── 메뉴 검색·레시피 파싱 도우미 ─────────────────────────
def secret(name, default=""):
    try:
        return st.secrets.get(name, default) or default
    except Exception:  # noqa: BLE001 — secrets 파일이 없으면 기본값
        return default


@st.cache_data
def get_public():
    return recipe_source.load_public()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_api(menu, key):
    return recipe_source.fetch_foodsafety(menu, key)


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def cached_ai(menu, key, model):
    return ai_estimate.estimate(menu, key, sorted(ING.index), model)


SERVING = ING["serving_g"].to_dict()


def parse_text(text, dish_name=None):
    return parser.parse_recipe_text(text, list(ING.index), SERVING, dish_name=dish_name)


def candidate_rows(cand):
    if cand["items"] is not None:
        return [{"원문": f"{n} {g:g}g", "재료": n, "양(g)": g, "상태": "자동"} for n, g in cand["items"]]
    return parse_text(cand["text"], cand["name"])


def render_parsed(rows, prefix, dish_label="", recipe_serv=1.0):
    df = pd.DataFrame(rows, columns=["원문", "재료", "양(g)", "상태"])
    cnt = parser.summarize(rows)
    st.caption(" · ".join(f"{k} {v}" for k, v in cnt.items() if v))
    if cnt.get("확인 필요"):
        st.warning("**확인 필요** 행은 DB에서 재료를 찾지 못했어요. '재료' 칸에서 가장 가까운 것을 고르거나 "
                   "(제외)를 선택하세요. 그대로 두면 판정에서 빠져 결과가 실제보다 좋게 나올 수 있어요.")
    ed = st.data_editor(df, key=f"{prefix}_ed", hide_index=True, use_container_width=True, disabled=["원문", "상태"],
                        column_config={
                            "재료": st.column_config.SelectboxColumn("재료(DB)", options=[parser.IGNORE] + sorted(ING.index)),
                            "양(g)": st.column_config.NumberColumn("양(g)", min_value=0.0, max_value=3000.0, step=5.0),
                        })
    c1, c2 = st.columns(2)
    rs = c1.number_input("이 레시피는 몇 인분?", 0.5, 20.0, float(recipe_serv), 0.5, key=f"{prefix}_rs")
    eat = c2.number_input("내가 먹은 양(인분)", 0.25, 5.0, 1.0, 0.25, key=f"{prefix}_eat")
    if st.button("➕ 이번 식사에 추가", type="primary", key=f"{prefix}_add", use_container_width=True):
        ok = ed[ed["재료"].notna() & (ed["재료"] != parser.IGNORE) & (ed["양(g)"].fillna(0) > 0)]
        ss.pending_add = [(r["재료"], round(float(r["양(g)"]) * eat / rs, 1)) for _, r in ok.iterrows()]
        ss["_pending_dish"] = dish_label


def tab_menu_search():
    fs_key, ai_key = secret("FOODSAFETY_API_KEY"), secret("ANTHROPIC_API_KEY")
    public = get_public()
    with st.form("menu_search", border=False):
        q = st.text_input("메뉴 이름", placeholder="예: 부대찌개, 감자탕, 크림파스타")
        go_search = st.form_submit_button("🔎 재료 찾기", use_container_width=True)
    if go_search and q.strip():
        q = q.strip()
        cands = recipe_source.search_builtin(q, REC) + recipe_source.search_public(q, public)
        if fs_key and not any(c["source"] == recipe_source.PUBLIC_SOURCE for c in cands):
            with st.spinner("공공 레시피 DB 검색 중..."):
                cands += cached_api(q, fs_key)
        ver = ss.get("search", {}).get("ver", 0) + 1
        ss["search"] = {"q": q, "cands": cands, "ver": ver, "ai": None}
    srch = ss.get("search")
    if not srch:
        src = ["앱 내장 레시피"] + (["식품안전나라 레시피 DB"] if public is not None or fs_key else []) + (["AI 추정"] if ai_key else [])
        st.caption("검색 출처: " + " → ".join(src) + ". 결과는 표에서 확인·수정한 뒤 추가돼요.")
        return
    if srch["cands"]:
        labels = [f"{c['name']}  ·  {c['source']}" for c in srch["cands"]]
        idx = st.radio("검색 결과", range(len(labels)), format_func=lambda i: labels[i], key=f"sr_{srch['ver']}")
        cand = srch["cands"][idx]
        if cand["text"]:
            with st.expander("레시피 원문 (재료)"):
                st.text(cand["text"])
                st.caption(f"출처: {cand['source']}" + (f" · 1인분 중량 {cand['weight']}" if cand.get("weight") else ""))
        render_parsed(candidate_rows(cand), f"s{srch['ver']}_{idx}", cand["name"], cand.get("servings", 1.0))
    else:
        st.info(f"'{srch['q']}' 레시피를 DB에서 찾지 못했어요." +
                ("" if ai_key else " 인터넷에서 찾은 레시피를 **📋 레시피 붙여넣기** 탭에 넣어 보세요."))
    if ai_key:
        st.divider()
        limit = int(secret("AI_SESSION_LIMIT", 30))
        used = ss.get("ai_calls", 0)
        if st.button(f"🤖 AI로 '{srch['q']}' 재료 추정하기", use_container_width=True, disabled=used >= limit,
                     key=f"ai_btn_{srch['ver']}"):
            try:
                with st.spinner("AI가 일반적인 1인분 레시피를 추정하는 중..."):
                    srch["ai"] = cached_ai(srch["q"], ai_key, secret("ANTHROPIC_MODEL", ai_estimate.DEFAULT_MODEL))
                ss["ai_calls"] = used + 1
            except Exception as e:  # noqa: BLE001
                st.error(f"AI 추정에 실패했어요 ({type(e).__name__}). 레시피 붙여넣기를 이용해 주세요.")
        st.caption(f"AI에는 메뉴 이름만 전송돼요 (식사·증상 기록은 보내지 않음). 이번 세션 사용 {used}/{limit}회")
        if srch.get("ai"):
            res = srch["ai"]
            if not res["ingredients"]:
                st.warning("AI가 이 메뉴를 알지 못했어요.")
            else:
                st.info(f"🤖 **AI 추정 레시피** — {res['note']}  \n가게·가정마다 크게 다를 수 있으니 반드시 확인·수정하세요.")
                rows = parse_text(ai_estimate.to_recipe_text(res))
                for r in rows:
                    if r["상태"] in ("자동", "추정"):
                        r["상태"] = "AI 추정"
                render_parsed(rows, f"ai{srch['ver']}", res.get("dish") or srch["q"])


def tab_paste():
    st.caption("블로그·레시피 사이트에서 **재료 부분**을 복사해 붙여 넣으면 g으로 바꿔 계산해요. "
               "'1큰술', '1/2개', '약간', '300g(1/4포기)' 같은 표현을 이해하고, 김치는 배추·마늘·고춧가루·파로 나눠요.")
    txt = st.text_area("레시피 재료", height=140,
                       placeholder="돼지고기 앞다리살 300g, 신김치 1/4포기, 양파 1/2개, 대파 1대,\n다진 마늘 1큰술, 고춧가루 2큰술, 국간장 1큰술, 물 3컵")
    name = st.text_input("메뉴 이름 (선택)", key="paste_name")
    if st.button("🧮 재료 계산", use_container_width=True) and txt.strip():
        ss["paste"] = {"rows": parse_text(txt), "ver": ss.get("paste", {}).get("ver", 0) + 1, "name": name}
    if ss.get("paste"):
        if not ss["paste"]["rows"]:
            st.warning("재료를 읽지 못했어요. '재료명 양' 형식인지 확인해 주세요.")
        else:
            render_parsed(ss["paste"]["rows"], f"p{ss['paste']['ver']}", ss["paste"]["name"])



def page_meal():
    st.title("🍽️ 식사 판정·기록")
    c = st.columns(3)
    d = c[0].date_input("날짜", value=date.today())
    t = c[1].time_input("시간", value=time(12, 30), step=600)
    label = c[2].selectbox("식사", ["아침", "점심", "저녁", "간식"], index=1)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("##### 1) 음식 추가")
        tab0, tabp, tab1, tab2 = st.tabs(["🔎 메뉴 검색", "📋 레시피 붙여넣기", "🍲 기본 요리", "🥕 재료"])
        with tab0:
            tab_menu_search()
        with tabp:
            tab_paste()
        with tab1:
            dish = st.selectbox("요리 선택", sorted(REC["dish"].unique()))
            serv = st.select_slider("인분", options=[0.5, 1.0, 1.5, 2.0], value=1.0)
            if st.button("➕ 요리 추가", use_container_width=True):
                ss.pending_add = db.expand_dish(REC, dish, serv)
                ss["_pending_dish"] = dish
            with st.expander("이 요리의 재료 구성 보기 (1인분 추정치)"):
                st.dataframe(REC[REC["dish"] == dish][["ingredient", "grams"]]
                             .rename(columns={"ingredient": "재료", "grams": "양(g)"}), hide_index=True)
                st.caption("가정·식당마다 레시피가 다르므로, 아래 표에서 양을 직접 고칠 수 있어요.")
        with tab2:
            names = sorted(ING.index)
            name = st.selectbox("재료 선택", names, index=names.index("흰쌀밥"))
            default = float(ING.loc[name, "serving_g"]) if not pd.isna(ING.loc[name, "serving_g"]) else 50.0
            grams = st.number_input("양 (g 또는 mL)", min_value=0.0, max_value=2000.0, value=default, step=5.0,
                                    key=f"g_{name}")
            if st.button("➕ 재료 추가", use_container_width=True):
                ss.pending_add = [(name, grams)]

        st.markdown("##### 2) 이번 식사 구성 (양 수정·행 삭제 가능)")
        edited = st.data_editor(
            ss.current_meal, key=f"meal_editor_{ss.editor_ver}", num_rows="dynamic", use_container_width=True,
            column_config={
                "ingredient": st.column_config.SelectboxColumn("재료", options=sorted(ING.index), required=True),
                "grams": st.column_config.NumberColumn("양(g)", min_value=0.0, max_value=2000.0, step=5.0, required=True),
            })
        if ss.pending_add:
            add = pd.DataFrame(ss.pending_add, columns=["ingredient", "grams"])
            base = edited.dropna(how="all")
            ss.current_meal = pd.concat([base, add], ignore_index=True) if not base.empty else add
            ss.pending_add = []
            ss.editor_ver += 1
            st.rerun()
        items_df = edited.dropna().query("grams > 0") if not edited.empty else edited

        b1, b2 = st.columns(2)
        if b1.button("💾 이 식사 기록하기", type="primary", use_container_width=True, disabled=items_df.empty):
            storage.add_meal(datetime.combine(d, t), label, items_df, ss.get("_pending_dish", ""))
            ss.current_meal = ss.current_meal.iloc[0:0]
            ss.editor_ver += 1
            ss["_toast"] = "식사를 기록했어요! ✅"
            st.rerun()
        if b2.button("🗑️ 비우기", use_container_width=True):
            ss.current_meal = ss.current_meal.iloc[0:0]
            ss.editor_ver += 1
            st.rerun()

    with right:
        st.markdown("##### 3) 판정 결과")
        if items_df.empty:
            st.info("왼쪽에서 요리나 재료를 추가하면 바로 판정해 드려요.")
        else:
            render_assessment(list(zip(items_df["ingredient"], items_df["grams"])))

    st.divider()
    st.subheader("최근 식사 기록")
    if ss.meal_items.empty:
        st.caption("아직 기록이 없어요.")
        return
    feats = analysis.meal_features(ss.meal_items, ING, LOOKUP).sort_values("datetime", ascending=False)
    view = feats.head(15).assign(
        일시=lambda x: x["datetime"].dt.strftime("%m/%d %H:%M"),
        판정=lambda x: x["overall"].map(db.STATUS_KO),
        재료=lambda x: x["ingredients"].str.replace("|", ", "),
    )[["일시", "meal_label", "판정", "재료"]].rename(columns={"meal_label": "식사"})
    st.dataframe(view, hide_index=True, use_container_width=True)
    with st.expander("기록 삭제"):
        opts = {r.meal_id: f"{r.datetime:%m/%d %H:%M} {r.meal_label} — {r.ingredients[:40]}" for r in feats.itertuples()}
        mid = st.selectbox("삭제할 식사", list(opts), format_func=opts.get)
        if st.button("삭제"):
            storage.delete_meal(mid)
            st.rerun()


# ───────────────────────── 증상·생활 기록 ─────────────────────────
def page_symptom():
    st.title("📝 증상·생활 기록")
    c1, c2 = st.columns(2, gap="large")
    with c1, st.form("sym_form", clear_on_submit=True):
        st.markdown("##### 증상 기록")
        a, b = st.columns(2)
        d = a.date_input("날짜", value=date.today())
        t = b.time_input("시간", value=datetime.now().time().replace(second=0, microsecond=0), step=600)
        pain = st.slider("복통", 0, 10, 0)
        bloat = st.slider("복부 팽만·가스", 0, 10, 0)
        dia = st.slider("설사·급박감", 0, 10, 0)
        con = st.slider("변비·잔변감", 0, 10, 0)
        note = st.text_input("메모 (선택)")
        if st.form_submit_button("기록하기", type="primary", use_container_width=True):
            storage.add_symptom(datetime.combine(d, t), pain, bloat, dia, con, note)
            ss["_toast"] = "증상을 기록했어요 📝"
            if max(pain, bloat, dia, con) >= 8:
                ss["_warn"] = "증상이 심해요. 참기 어렵거나 열·혈변·구토가 동반되면 바로 진료를 받으세요."
            st.rerun()
    with c2, st.form("daily_form"):
        st.markdown("##### 하루 생활 요인 (교란 요인 보정용)")
        d2 = st.date_input("날짜 ", value=date.today())
        stress = st.slider("스트레스", 0, 10, 4)
        sleep = st.number_input("수면 시간", 0.0, 14.0, 7.0, 0.5)
        mens = st.checkbox("월경 기간")
        if st.form_submit_button("저장", use_container_width=True):
            storage.upsert_daily(d2, stress, sleep, mens)
            ss["_toast"] = "생활 요인을 저장했어요 ✅"
            st.rerun()
        st.caption("음식 말고도 스트레스·수면 부족·월경 주기가 IBS 증상에 영향을 줘요. "
                   "같이 기록하면 '음식 탓'을 잘못 짚을 가능성이 줄어요.")
    st.info("💡 **증상이 없었던 날도 0점으로 기록해 주세요.** 기록이 없으면 '증상 없음'으로 간주하므로, "
            "빠뜨린 증상이 있으면 분석이 부정확해져요.")

    if not ss.symptoms.empty:
        st.subheader("증상 추이")
        s = analysis.symptom_scores(ss.symptoms).sort_values("datetime")
        fig = px.bar(s, x="datetime", y="score", color_discrete_sequence=["#d64545"],
                     labels={"datetime": "", "score": "최고 증상 점수"})
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(s.sort_values("datetime", ascending=False).head(20)
                     .rename(columns={"datetime": "일시", "pain": "복통", "bloating": "팽만", "diarrhea": "설사",
                                      "constipation": "변비", "note": "메모", "score": "최고점"}),
                     hide_index=True, use_container_width=True)


# ───────────────────────── 트리거 분석 ─────────────────────────
def page_analysis():
    st.title("📊 나의 트리거 분석")
    if ss.meal_items.empty:
        st.info("기록이 없어요. 🏠 홈에서 **가상 사용자 데이터**를 불러와 체험해 보세요.")
        return
    with st.expander("⚙️ 분석 설정", expanded=False):
        lag = st.slider("식후 몇 시간 사이의 증상을 그 식사와 연결할까요?", 0, 24, (1, 8))
        thr = st.slider("몇 점 이상을 '증상 있음'으로 볼까요?", 1, 10, 4)
        min_n = st.slider("비교에 필요한 최소 식사 수", 3, 15, 5)
        st.caption("기본값(식후 1~8시간, 4점 이상, 최소 5끼)은 가상 환자 검증에서 선택한 값입니다.")
    df = dataset(lag[0], lag[1], thr)
    days = df["datetime"].dt.date.nunique()

    m = st.columns(4)
    m[0].metric("기록 일수", f"{days}일")
    m[1].metric("식사 수", f"{len(df)}끼")
    m[2].metric("식후 증상 동반", f"{df['outcome'].mean():.0%}")
    m[3].metric("고포드맵 식사 비율", f"{(df['overall'] == 'high').mean():.0%}")
    if days < 14:
        st.warning("기록이 2주 미만이에요. 검증 결과, 14일 미만에서는 트리거를 찾을 확률이 낮아요 (🧪 알고리즘 검증 참고).")
    if ss.demo_info:
        st.info("🧪 데모 정답: 이 가상 사용자는 유당과 매운맛(캡사이신)에 민감하도록 설정됐어요. 앱이 맞혔나요?")

    tt = analysis.trigger_table(df, min_n=min_n)
    if tt.empty:
        st.info("분석할 데이터가 부족해요.")
        return

    st.subheader("요약")
    hits = tt[tt["판정"].isin(["의심 높음", "가능성 있음"])]
    if hits.empty:
        st.success("아직 뚜렷하게 의심되는 성분이 없어요. 기록을 계속 쌓아 주세요.")
    for _, r in hits.iterrows():
        color = LABEL_COLOR[r["판정"]]
        st.markdown(
            f"<div style='padding:10px 14px;margin-bottom:8px;border-left:6px solid {color};background:{color}14;border-radius:6px'>"
            f"<b style='color:{color}'>{r['판정']}</b> · <b>{r['이름']}</b><br>"
            f"{r['이름']}이(가) 많은 식사 후 증상 <b>{r['rate1']:.0%}</b> ({r['k1']}/{r['n1']}끼) vs "
            f"적은 식사 후 <b>{r['rate0']:.0%}</b> ({r['k0']}/{r['n0']}끼) → 약 <b>{r['rr']:.1f}배</b> "
            f"(95% 구간 {r['rr_lo']:.1f}~{r['rr_hi']:.1f}) · <span style='color:#777'>{r['단계']}</span></div>",
            unsafe_allow_html=True)
    blocked = tt[tt["판정"] == "비교 불가(거의 매번 섭취)"]
    if not blocked.empty:
        st.info("🔁 **" + ", ".join(blocked["이름"]) + "**: 거의 매 끼니 먹고 있어서 비교할 '안 먹은 식사'가 부족해요. "
                "한국 식단에서는 특히 마늘·양파의 **프럭탄**이 그래요. 이 경우 **제한기 → 재도입 챌린지**로 확인하는 것이 정확해요.")

    st.subheader("위험비(RR) 한눈에 보기")
    plot = tt[~tt["판정"].isin(["데이터 부족", "비교 불가(거의 매번 섭취)"])].copy()
    if not plot.empty:
        fig = go.Figure()
        for lab, g in plot.groupby("판정"):
            fig.add_trace(go.Scatter(
                x=g["rr"], y=g["이름"], mode="markers", name=lab, marker=dict(size=12, color=LABEL_COLOR[lab]),
                error_x=dict(type="data", symmetric=False, array=g["rr_hi"] - g["rr"], arrayminus=g["rr"] - g["rr_lo"]),
                hovertemplate="%{y}: RR %{x:.2f}<extra></extra>"))
        fig.add_vline(x=1, line_dash="dash", line_color="#888")
        fig.update_layout(height=60 + 38 * len(plot), xaxis_type="log", xaxis_title="위험비 (1보다 크면 섭취 시 증상↑, 로그 축)",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25),
                          yaxis=dict(categoryorder="array", categoryarray=list(plot["이름"])[::-1]))
        st.plotly_chart(fig, use_container_width=True)

    show = tt[["이름", "종류", "판정", "n1", "k1", "n0", "k0", "rate1", "rate0", "rr", "rr_lo", "rr_hi", "p_greater", "단계"]]
    st.dataframe(show.rename(columns={"n1": "노출 식사", "k1": "노출 후 증상", "n0": "비노출 식사", "k0": "비노출 후 증상",
                                      "rate1": "노출 시 증상률", "rate0": "비노출 시 증상률", "rr": "위험비", "rr_lo": "95% 하한",
                                      "rr_hi": "95% 상한", "p_greater": "P(노출>비노출)"}),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                ["노출 시 증상률", "비노출 시 증상률", "위험비", "95% 하한", "95% 상한", "P(노출>비노출)"]})

    with st.expander("🔎 분석 방법 설명"):
        st.markdown(f"""
- **관찰 단위 = 식사 1끼.** 성분 부하가 {db.EXPOSE_LOAD} 이상이면 '노출', 식후 {lag[0]}~{lag[1]}시간 안에 {thr}점 이상 증상이 있으면 '증상 있음'.
- **베이지안 비교:** 노출·비노출 식사의 증상률에 Beta(1,1) 사전분포를 두고 4,000번 표본추출 → 위험비와 95% 신용구간, P(노출 시 증상률 > 비노출 시 증상률).
- **판정:** P ≥ {analysis.P_HIGH} → 의심 높음, ≥ {analysis.P_MID} → 가능성 있음. 10개 요인을 동시에 보므로(다중비교) 0.95보다 엄격하게 잡았어요.
- **단계적 조건부 검정:** 가장 강한 요인을 확정한 뒤, 그 요인이 *없는* 식사만으로 나머지를 다시 비교해요. 김치찌개처럼 마늘·고춧가루가 함께 들어가 덩달아 의심받는 문제를 줄여요.
- **한계:** 연관성일 뿐 인과관계가 아니에요. 기록 누락, 레시피 추정 오차, 시간차 가정이 결과에 영향을 줘요. 최종 확인은 **재도입 챌린지**로 하세요.
""")
    with st.expander("🥕 재료 단위 분석 (보조)"):
        st.caption("마늘·양파·대파처럼 함께 쓰이는 재료는 서로 구분이 어려우니 참고만 하세요.")
        it = analysis.ingredient_table(df, min_n=min_n)
        if not it.empty:
            st.dataframe(it[["재료", "판정", "n1", "k1", "rate1", "rate0", "rr", "p_greater"]].head(20), hide_index=True,
                         use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["rate1", "rate0", "rr", "p_greater"]})
    with st.expander("📐 교란요인 보정 모델 (로지스틱 회귀)"):
        st.caption("스트레스·수면·월경을 함께 넣어 각 요인의 독립적인 효과(오즈비)를 추정합니다. 부트스트랩 200회로 95% 구간 계산.")
        if st.button("보정 모델 계산"):
            with st.spinner("계산 중..."):
                adj = analysis.adjusted_model(df)
            if adj is None:
                st.info("데이터가 부족해서 계산할 수 없어요 (최소 20끼, 증상 있음/없음 모두 필요).")
            else:
                st.dataframe(adj, hide_index=True, use_container_width=True,
                             column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["오즈비(OR)", "95% 하한", "95% 상한"]})

    st.subheader("타임라인")
    sym = analysis.symptom_scores(ss.symptoms) if not ss.symptoms.empty else pd.DataFrame(columns=["datetime", "score"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sym["datetime"], y=sym["score"], name="증상 점수", marker_color="#d64545", opacity=0.6))
    for st_, g in df.groupby("overall"):
        fig.add_trace(go.Scatter(x=g["datetime"], y=[-0.8] * len(g), mode="markers", name=f"식사 {db.STATUS_KO[st_]}",
                                 marker=dict(color=STATUS_COLOR[st_], size=9, symbol="triangle-up"),
                                 text=g["ingredients"].str.replace("|", ", "), hovertemplate="%{x}<br>%{text}<extra></extra>"))
    fig.update_layout(height=320, yaxis=dict(range=[-1.6, 10.5], title="증상 점수"), margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


# ───────────────────────── 재도입 챌린지 ─────────────────────────
def page_challenge():
    st.title("🔁 재도입 챌린지")
    st.markdown("""
저포드맵 식단은 **평생 제한식이 아니에요.** 표준 프로토콜은 3단계입니다.
1. **제한기 (2~6주)**: 고포드맵을 줄여 증상이 좋아지는지 확인
2. **재도입기 (6~10주)**: 성분을 *하나씩* 다시 먹어 보며 무엇에 민감한지 확인 ← 이 화면
3. **개인화**: 견디는 성분은 다시 먹고, 민감한 성분만 견딜 수 있는 양으로 조절

장기간 과도하게 제한하면 장내 미생물 다양성이 줄고 영양이 불균형해질 수 있어요.
""")
    st.warning("교육용 예시입니다. 실제 재도입은 의사·영양사와 일정을 정해 진행하세요.", icon="⚠️")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        sg = st.selectbox("테스트할 성분", list(challenge.PLANS), format_func=fmt_sg)
        plan = challenge.PLANS[sg]
        start = st.date_input("시작일", value=date.today())
        sched = pd.DataFrame({
            "날짜": [(start + timedelta(days=i)).strftime("%m/%d (%a)") for i in range(6)],
            "할 일": [f"{plan['food']} {plan['doses'][0]}", f"{plan['food']} {plan['doses'][1]}",
                     f"{plan['food']} {plan['doses'][2]}", "휴지기 (제한식 유지)", "휴지기", "휴지기 → 다음 성분"],
        })
        st.dataframe(sched, hide_index=True, use_container_width=True)
        st.caption("테스트 기간에는 다른 고포드맵 음식은 계속 제한해야 결과를 해석할 수 있어요.")
    with c2:
        baseline = st.slider("제한기 동안의 평소 증상 점수 (기저)", 0, 10, 2)
        scores = []
        for i in range(3):
            done = st.checkbox(f"{i + 1}일차 완료", key=f"ch_done_{sg}_{i}")
            scores.append(st.slider(f"{i + 1}일차 최고 증상 점수", 0, 10, baseline, key=f"ch_s_{sg}_{i}") if done else None)
        res = challenge.evaluate(baseline, scores)
        color = {"반응 있음": "#d64545", "내성 양호": "#2e9d5b", "진행 중": "#5b7db1"}[res["result"]]
        st.markdown(f"<div style='padding:12px;border-radius:10px;border:2px solid {color}'>"
                    f"<b style='color:{color};font-size:1.2rem'>{res['result']}</b><br>{res['message']}</div>",
                    unsafe_allow_html=True)
        if res["result"] != "진행 중" and st.button("결과 저장", type="primary"):
            ss.challenges.append({"성분": fmt_sg(sg), "시작일": start.isoformat(), "결과": res["result"],
                                  "반응 용량": plan["doses"][res["day"] - 1] if res["day"] else "-"})
            st.toast("저장했어요")
    if ss.challenges:
        st.subheader("나의 내성 프로필")
        st.dataframe(pd.DataFrame(ss.challenges), hide_index=True, use_container_width=True)


# ───────────────────────── 알고리즘 검증 ─────────────────────────
def page_validation():
    st.title("🧪 알고리즘 검증 (가상 환자 시뮬레이션)")
    st.markdown("""
실제 환자 데이터 없이 알고리즘을 검증하기 위해, **정답 트리거를 미리 정해 둔 가상 환자 300명**의 식사·증상 일지를 만들고
앱의 분석이 정답을 얼마나 맞히는지 측정했습니다. (`validation/run_validation.py`로 누구나 재현 가능)

- 가상 환자의 식단은 이 앱의 한국 요리 DB에서 무작위로 고르고, 정답 성분 부하가 클수록 식후 2~7시간에 증상이 날 확률이 커지도록 설정
- 15%는 트리거가 없는 **대조 환자**(배경 증상만) → 거짓 경보 측정
- **v1(초기)**: 각 요인을 따로 P≥0.95로 판정 → **v2(최종)**: P≥0.99 + 단계적 조건부 검정
""")
    p = ROOT / "validation" / "results.csv"
    if not p.exists():
        st.error("validation/results.csv가 없어요. `python validation/run_validation.py`를 먼저 실행하세요.")
        return
    r = pd.read_csv(p)
    c1, c2 = st.columns(2)
    long = pd.concat([
        pd.DataFrame({"기록 일수": r["days"], "값": r["민감도"], "지표": "민감도", "버전": "v2(최종)"}),
        pd.DataFrame({"기록 일수": r["days"], "값": r["정밀도"], "지표": "정밀도", "버전": "v2(최종)"}),
        pd.DataFrame({"기록 일수": r["days"], "값": r["v1 민감도"], "지표": "민감도", "버전": "v1(초기)"}),
        pd.DataFrame({"기록 일수": r["days"], "값": r["v1 정밀도"], "지표": "정밀도", "버전": "v1(초기)"}),
    ])
    fig = px.line(long, x="기록 일수", y="값", color="지표", line_dash="버전", markers=True,
                  color_discrete_map={"민감도": "#3a6fd8", "정밀도": "#2e9d5b"})
    fig.update_layout(height=340, yaxis_range=[0, 1], yaxis_tickformat=".0%", title="민감도·정밀도", margin=dict(t=40))
    c1.plotly_chart(fig, use_container_width=True)
    fp = pd.concat([
        pd.DataFrame({"기록 일수": r["days"], "환자당 거짓 경보": r["환자당 거짓경보"], "버전": "v2(최종)"}),
        pd.DataFrame({"기록 일수": r["days"], "환자당 거짓 경보": r["v1 환자당 거짓경보"], "버전": "v1(초기)"}),
    ])
    fig2 = px.bar(fp, x="기록 일수", y="환자당 거짓 경보", color="버전", barmode="group",
                  color_discrete_map={"v1(초기)": "#b8bcc4", "v2(최종)": "#d64545"})
    fig2.update_layout(height=340, title="거짓 경보 (적을수록 좋음)", margin=dict(t=40))
    c2.plotly_chart(fig2, use_container_width=True)

    hit = pd.concat([
        pd.DataFrame({"기록 일수": r["days"], "1순위 적중률": r["1순위 적중률(본 방법)"], "방법": "본 앱(베이지안 비교)"}),
        pd.DataFrame({"기록 일수": r["days"], "1순위 적중률": r["1순위 적중률(단순 빈도법)"], "방법": "단순 빈도법"}),
    ])
    fig3 = px.line(hit, x="기록 일수", y="1순위 적중률", color="방법", markers=True,
                   color_discrete_map={"본 앱(베이지안 비교)": "#3a6fd8", "단순 빈도법": "#b8bcc4"})
    fig3.update_layout(height=300, yaxis_range=[0, 1], yaxis_tickformat=".0%", margin=dict(t=30),
                       title="'가장 의심되는 1개'가 정답일 확률")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("단순 빈도법 = '증상이 있던 식사에 가장 자주 들어 있던 성분'. 한국 식단은 거의 모든 끼니에 프럭탄이 있어서 "
               "이 방법은 대부분 프럭탄을 지목해 버립니다.")

    pf = ROOT / "validation" / "by_feature.csv"
    if pf.exists():
        st.subheader("성분별 탐지율 (민감도)")
        bf = pd.read_csv(pf, index_col=0)
        st.dataframe(bf, use_container_width=True,
                     column_config={c: st.column_config.ProgressColumn(f"{c}일", min_value=0, max_value=1, format="percent")
                                    for c in bf.columns})
        st.caption("만니톨처럼 일상 식단에 거의 없는 성분은 기록만으로는 찾기 어렵습니다 → 재도입 챌린지가 필요한 이유.")
    st.subheader("전체 결과표")
    st.dataframe(r.round(3), hide_index=True, use_container_width=True)
    st.info("**해석:** 4주 기록 시 '의심 높음' 판정의 약 80%가 실제 정답(정밀도)이고, 정답 트리거의 약 40%를 찾아냅니다(민감도). "
            "보수적으로 설계해 거짓 경보를 줄인 대신, 놓친 트리거는 재도입 챌린지로 보완하도록 했습니다. "
            "단, 이 결과는 **가정한 증상 모델 위에서의 성능**이며 실제 환자에서의 성능은 아닙니다.")


# ───────────────────────── 데이터·출처 ─────────────────────────
def page_data():
    st.title("📚 데이터·출처")
    st.warning("""**포드맵 수치에 대한 중요한 안내**
이 앱의 식품별 기준값(🟢 최대량 / 🔴 최소량)은 공개된 학술 문헌과 공개 자료를 바탕으로 팀이 정리한 **교육용 근사치**입니다.
Monash University FODMAP 앱의 데이터베이스는 상업용이므로 복제하지 않았습니다.
`confidence` 열: H=여러 공개 자료에서 일관됨 / M=근거 있으나 수치 불확실 / L=한국 식품이라 직접 측정 자료가 부족해 레시피 재료로 추정.""")
    tab1, tab2, tab3 = st.tabs(["재료 DB", "요리 레시피", "참고문헌"])
    with tab1:
        q = st.text_input("검색", placeholder="예: 마늘")
        cat = st.multiselect("분류", sorted(ING["category"].unique()))
        v = ING.reset_index(drop=True)
        if q:
            v = v[v["name"].str.contains(q)]
        if cat:
            v = v[v["category"].isin(cat)]
        st.dataframe(v.rename(columns={"name": "재료", "category": "분류", "subgroups": "포드맵 성분", "green_max_g": "🟢 최대(g)",
                                       "red_min_g": "🔴 최소(g)", "serving_g": "1회량(g)", "flags": "기타 요인", "swap": "대체 제안"}),
                     hide_index=True, use_container_width=True)
        st.caption(f"재료 {len(ING)}개")
    with tab2:
        dish = st.selectbox("요리", sorted(REC["dish"].unique()), key="data_dish")
        items = db.expand_dish(REC, dish)
        st.dataframe(pd.DataFrame(items, columns=["재료", "1인분(g)"]), hide_index=True)
        st.caption(f"요리 {REC['dish'].nunique()}개 · 1인분 재료량은 일반적인 가정식 레시피를 참고한 팀 추정치입니다.")
    with tab3:
        st.markdown((ROOT / "docs" / "references.md").read_text(encoding="utf-8"))


if ss.get("_toast"):
    st.toast(ss.pop("_toast"))
if ss.get("_warn"):
    st.warning(ss.pop("_warn"), icon="🚨")

{"🏠 홈": page_home, "🍽️ 식사 판정·기록": page_meal, "📝 증상·생활 기록": page_symptom, "📊 트리거 분석": page_analysis,
 "🔁 재도입 챌린지": page_challenge, "🧪 알고리즘 검증": page_validation, "📚 데이터·출처": page_data}[page]()
