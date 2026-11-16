import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI 작업자 안전 모니터링",
    page_icon="🦺",
    layout="wide"
)

# ----------------------------
# Style
# ----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(120,120,120,0.28);
    border-radius: 14px;
    padding: 16px 18px;
    background: rgba(120,120,120,0.06);
}
.hero-box {
    padding: 22px 24px;
    border-radius: 16px;
    border: 1px solid rgba(120,120,120,0.28);
    background: rgba(120,120,120,0.06);
    margin-bottom: 18px;
}
.small-note {
    color: #8b8b8b;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.title("🦺 AI 작업자 안전 모니터링")
st.caption("MediaPipe Pose 기반 관절 움직임 분석 · Streamlit 대시보드")

st.markdown("""
<div class="hero-box">
<b>서비스 목적</b><br>
작업자의 어깨·팔꿈치·손목 관절 움직임과 위험구역 진입 정보를 바탕으로
NORMAL / CAUTION / DANGER 상태를 시각적으로 확인하는 학습용 프로토타입입니다.<br><br>
<span class="small-note">※ 실제 산업재해 발생 가능성을 예측하거나 사고 검출 정확도를 보장하는 시스템은 아닙니다.</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Demo data
# ----------------------------
demo = pd.DataFrame({
    "동작": ["B 정상 통과", "C 급정지", "D 팔꿈치 움직임"],
    "구역 내 프레임": [259, 631, 992],
    "Baseline DANGER": [259, 631, 992],
    "제안 방법 DANGER": [7, 21, 85]
})
demo["Baseline DANGER 비율"] = demo["Baseline DANGER"] / demo["구역 내 프레임"] * 100
demo["제안 방법 DANGER 비율"] = demo["제안 방법 DANGER"] / demo["구역 내 프레임"] * 100

TOTAL_FRAMES = 3598
IN_ZONE_FRAMES = 1882
DANGER_FRAMES = int(demo["제안 방법 DANGER"].sum())
CAUTION_FRAMES = IN_ZONE_FRAMES - DANGER_FRAMES
NORMAL_FRAMES = TOTAL_FRAMES - IN_ZONE_FRAMES

sample_csv = demo[["동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"]].to_csv(index=False).encode("utf-8-sig")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("데이터 선택")
    mode = st.radio(
        "분석할 데이터를 선택하세요.",
        ["예제 데이터", "CSV 업로드"],
        index=0
    )

    uploaded_file = None
    if mode == "CSV 업로드":
        uploaded_file = st.file_uploader(
            "분석 결과 CSV 업로드",
            type=["csv"]
        )
        st.caption(
            "동작별 요약 CSV 또는 condition / in_zone / status 열이 포함된 "
            "프레임 단위 CSV를 지원합니다."
        )

    st.divider()
    st.download_button(
        "📥 예제 CSV 다운로드",
        data=sample_csv,
        file_name="sample_safety_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()
    st.markdown("### 사용 방법")
    st.markdown(
        "1. 예제 데이터로 대시보드를 확인합니다.\n"
        "2. 필요하면 CSV 업로드를 선택합니다.\n"
        "3. KPI, 그래프, 표와 결과 해석을 확인합니다."
    )

def normalize_condition_name(value):
    text = str(value).strip()
    mapping = {
        "A": "A 구역 밖 정상",
        "B": "B 정상 통과",
        "C": "C 급정지",
        "D": "D 팔꿈치 움직임",
    }
    return mapping.get(text, text)

def build_summary_from_raw(df):
    condition_candidates = ["condition", "Condition", "동작", "scenario"]
    zone_candidates = ["in_zone", "zone", "위험구역", "구역 내", "inside_zone"]
    status_candidates = ["status", "risk_status", "판정", "state"]

    cond_col = next((c for c in condition_candidates if c in df.columns), None)
    zone_col = next((c for c in zone_candidates if c in df.columns), None)
    status_col = next((c for c in status_candidates if c in df.columns), None)

    if not (cond_col and zone_col and status_col):
        return None

    work = df[[cond_col, zone_col, status_col]].copy()
    work[cond_col] = work[cond_col].apply(normalize_condition_name)

    zone_values = work[zone_col]
    if zone_values.dtype == object:
        zone_bool = zone_values.astype(str).str.lower().isin(
            ["1", "true", "t", "yes", "y", "in", "inside"]
        )
    else:
        zone_bool = pd.to_numeric(zone_values, errors="coerce").fillna(0) > 0

    work["_in_zone"] = zone_bool
    work["_danger"] = work[status_col].astype(str).str.upper().eq("DANGER")

    rows = []
    for cond, group in work.groupby(cond_col):
        in_zone = int(group["_in_zone"].sum())
        if in_zone == 0:
            continue
        danger = int((group["_in_zone"] & group["_danger"]).sum())
        rows.append({
            "동작": cond,
            "구역 내 프레임": in_zone,
            "Baseline DANGER": in_zone,
            "제안 방법 DANGER": danger,
        })

    if not rows:
        return None

    out = pd.DataFrame(rows)
    out["Baseline DANGER 비율"] = 100.0
    out["제안 방법 DANGER 비율"] = out["제안 방법 DANGER"] / out["구역 내 프레임"] * 100
    return out

def build_summary_from_aggregate(df):
    ko_cols = ["동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"]
    en_cols = ["condition", "in_zone_frames", "baseline_danger", "proposed_danger"]

    if set(ko_cols).issubset(df.columns):
        out = df[ko_cols].copy()
    elif set(en_cols).issubset(df.columns):
        out = df[en_cols].copy()
        out.columns = ko_cols
        out["동작"] = out["동작"].apply(normalize_condition_name)
    else:
        return None

    for c in ["구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna()
    if out.empty:
        return None

    out["Baseline DANGER 비율"] = out["Baseline DANGER"] / out["구역 내 프레임"] * 100
    out["제안 방법 DANGER 비율"] = out["제안 방법 DANGER"] / out["구역 내 프레임"] * 100
    return out

summary = demo.copy()
raw_df = None
using_demo = True

if mode == "CSV 업로드":
    if uploaded_file is None:
        st.info("왼쪽 사이드바에서 CSV 파일을 업로드하면 해당 데이터로 자동 분석합니다.")
    else:
        try:
            raw_df = pd.read_csv(uploaded_file)
            parsed = build_summary_from_aggregate(raw_df)
            if parsed is None:
                parsed = build_summary_from_raw(raw_df)

            if parsed is None:
                st.warning(
                    "CSV 열 구조를 자동으로 인식하지 못해 예제 데이터를 표시합니다. "
                    "업로드한 CSV의 열 이름을 확인해 주세요."
                )
                st.write("업로드한 CSV 열:", list(raw_df.columns))
            else:
                summary = parsed
                using_demo = False
                st.success("✅ 업로드한 CSV를 분석했습니다.")
        except Exception as e:
            st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")

# ----------------------------
# Current mode
# ----------------------------
mode_label = "예제 실험 데이터" if using_demo else "업로드 데이터"
st.markdown(f"**현재 분석 데이터:** {mode_label}")

# ----------------------------
# KPI
# ----------------------------
if using_demo:
    total_frames = TOTAL_FRAMES
    in_zone_frames = IN_ZONE_FRAMES
    danger_frames = DANGER_FRAMES
    caution_frames = CAUTION_FRAMES
    normal_frames = NORMAL_FRAMES
else:
    in_zone_frames = int(summary["구역 내 프레임"].sum())
    danger_frames = int(summary["제안 방법 DANGER"].sum())
    caution_frames = max(in_zone_frames - danger_frames, 0)
    total_frames = len(raw_df) if raw_df is not None else in_zone_frames
    normal_frames = max(total_frames - in_zone_frames, 0)

st.subheader("1. 위험 상태 요약")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("전체 프레임", f"{total_frames:,}")
c2.metric("NORMAL", f"{normal_frames:,}")
c3.metric("위험구역 진입", f"{in_zone_frames:,}")
c4.metric("CAUTION", f"{caution_frames:,}")
c5.metric("DANGER", f"{danger_frames:,}")

st.divider()

# ----------------------------
# Overall state + action danger
# ----------------------------
st.subheader("2. 상태 분포 및 동작별 분석")
left, right = st.columns([0.85, 1.15])

with left:
    state_df = pd.DataFrame({
        "상태": ["NORMAL", "CAUTION", "DANGER"],
        "프레임": [normal_frames, caution_frames, danger_frames]
    })
    donut = px.pie(
        state_df,
        names="상태",
        values="프레임",
        hole=0.58,
        title="전체 위험 상태 분포",
        color="상태",
        color_discrete_map={
            "NORMAL": "#4CAF50",
            "CAUTION": "#F2B84B",
            "DANGER": "#E45756"
        }
    )
    donut.update_traces(textinfo="percent+label")
    donut.update_layout(margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(donut, use_container_width=True)

with right:
    bar = summary[["동작", "제안 방법 DANGER 비율"]].copy()
    fig = px.bar(
        bar,
        x="동작",
        y="제안 방법 DANGER 비율",
        text=bar["제안 방법 DANGER 비율"].map(lambda x: f"{x:.1f}%"),
        labels={"제안 방법 DANGER 비율": "DANGER 비율 (%)"},
        title="동작별 DANGER 비율"
    )
    fig.update_traces(textposition="outside", marker_color="#5B8FF9")
    fig.update_layout(
        yaxis_range=[0, max(12, float(bar["제안 방법 DANGER 비율"].max()) * 1.35)],
        showlegend=False,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Baseline comparison
# ----------------------------
st.subheader("3. Baseline 대비 제안 방법")

compare = summary[[
    "동작", "Baseline DANGER 비율", "제안 방법 DANGER 비율"
]].melt(
    id_vars="동작",
    var_name="방법",
    value_name="DANGER 비율"
)
compare["방법"] = compare["방법"].replace({
    "Baseline DANGER 비율": "Baseline",
    "제안 방법 DANGER 비율": "제안 방법",
})

fig2 = px.bar(
    compare,
    x="동작",
    y="DANGER 비율",
    color="방법",
    barmode="group",
    text=compare["DANGER 비율"].map(lambda x: f"{x:.1f}%"),
    labels={"DANGER 비율": "DANGER 비율 (%)"},
    color_discrete_map={"Baseline": "#A9C9FF", "제안 방법": "#315EBD"}
)
fig2.update_traces(textposition="outside")
fig2.update_layout(
    yaxis_range=[0, 112],
    legend_title_text="방법",
    margin=dict(t=30, b=20, l=20, r=20)
)
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Table
# ----------------------------
st.subheader("4. 동작별 분석 결과")

display = summary[[
    "동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER",
    "제안 방법 DANGER 비율"
]].copy()
display["제안 방법 DANGER 비율"] = display["제안 방법 DANGER 비율"].map(lambda x: f"{x:.1f}%")
st.dataframe(display, use_container_width=True, hide_index=True)

# ----------------------------
# Interpretation
# ----------------------------
st.subheader("5. 결과 해석")

if len(summary) > 0:
    max_row = summary.loc[summary["제안 방법 DANGER 비율"].idxmax()]
    st.markdown(
        f"- 현재 데이터에서는 **{max_row['동작']}**에서 "
        f"가장 높은 DANGER 비율 **{max_row['제안 방법 DANGER 비율']:.1f}%**가 나타났습니다."
    )

if using_demo:
    st.markdown(
        "- 정상 통과 **B(2.7%)**와 급정지 **C(3.3%)**의 차이는 **0.6%p**로 작아, "
        "현재 규칙만으로 급정지 동작을 명확하게 구분하는 데 한계가 있습니다."
    )
    st.markdown(
        "- **D 팔꿈치 움직임은 8.6%**로 B와 C보다 높은 DANGER 비율을 보였습니다."
    )

st.warning(
    "주의: 본 결과는 모의 실험의 프레임 단위 상태 분포이며, "
    "실제 끼임 사고의 발생 가능성이나 사고 검출 정확도를 의미하지 않습니다."
)

# ----------------------------
# Raw preview
# ----------------------------
if raw_df is not None and not using_demo:
    with st.expander("업로드 데이터 미리보기"):
        st.dataframe(raw_df.head(30), use_container_width=True)

with st.expander("서비스 및 분석 방법"):
    st.markdown(
        """
        **분석 흐름**
        1. 웹캠 영상에서 MediaPipe Pose로 어깨·팔꿈치·손목 관절 좌표를 추정합니다.
        2. 위험구역 진입 여부와 손목 이동속도, 팔꿈치 각도 변화를 계산합니다.
        3. 규칙 기반 위험점수를 이용해 NORMAL / CAUTION / DANGER로 구분합니다.
        4. Streamlit 대시보드에서 동작별 결과를 시각화합니다.

        **실험 조건**
        - 참가자 1명이 A~D 모의 동작을 각각 4회 수행했습니다.
        - 현재 저장 데이터에는 반복별 시작·종료 구분이 없어 프레임 단위 비율을 사용합니다.

        **현재 버전의 범위**
        - 수업 과제를 위해 제작한 학습용 프로토타입입니다.
        - 실제 산업현장 적용 전에는 다양한 작업자와 환경에서 추가 검증이 필요합니다.
        """
    )
