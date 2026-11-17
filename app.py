import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="SafePose AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2.0rem;
    padding-bottom: 3.5rem;
}
.hero-kicker {
    display: inline-block;
    padding: 0.32rem 0.7rem;
    border-radius: 999px;
    background: #EAF2FF;
    color: #2457A7;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}
.badge-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
    margin-bottom: 0.7rem;
}
.badge {
    padding: 0.30rem 0.62rem;
    border: 1px solid #D7E3F4;
    border-radius: 999px;
    color: #40546F;
    background: #F8FBFF;
    font-size: 0.80rem;
    font-weight: 600;
}
.feature-card {
    border: 1px solid #DFE6EF;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    background: #FFFFFF;
    min-height: 105px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}
.feature-label {
    font-size: 0.82rem;
    color: #6B778C;
    margin-bottom: 0.35rem;
    font-weight: 700;
}
.feature-value {
    font-size: 1.0rem;
    color: #172033;
    font-weight: 700;
}
.kpi-card {
    border: 1px solid #E0E6EE;
    border-top: 4px solid #4F78D1;
    border-radius: 14px;
    padding: 0.9rem 1rem 0.95rem 1rem;
    background: #FFFFFF;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}
.kpi-label {
    color: #697586;
    font-size: 0.80rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.kpi-value {
    color: #172033;
    font-size: 2rem;
    line-height: 1.05;
    font-weight: 700;
}
.kpi-total { border-top-color: #4F78D1; }
.kpi-normal { border-top-color: #22A06B; }
.kpi-zone { border-top-color: #7C63D9; }
.kpi-caution { border-top-color: #E2A400; }
.kpi-danger { border-top-color: #E5484D; }
.section-note {
    border-left: 4px solid #4F78D1;
    background: #F7FAFE;
    padding: 0.75rem 0.95rem;
    border-radius: 8px;
    color: #445065;
    margin: 0.4rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data
# ----------------------------
demo = pd.DataFrame({
    "동작": ["B 정상 통과", "C 급정지", "D 팔꿈치 움직임"],
    "구역 내 프레임": [259, 631, 992],
    "Baseline DANGER": [259, 631, 992],
    "제안 방법 DANGER": [7, 21, 85],
})
demo["Baseline DANGER 비율"] = demo["Baseline DANGER"] / demo["구역 내 프레임"] * 100
demo["제안 방법 DANGER 비율"] = demo["제안 방법 DANGER"] / demo["구역 내 프레임"] * 100

TOTAL_FRAMES = 3598
IN_ZONE_FRAMES = 1882
DANGER_FRAMES = int(demo["제안 방법 DANGER"].sum())
CAUTION_FRAMES = IN_ZONE_FRAMES - DANGER_FRAMES
NORMAL_FRAMES = TOTAL_FRAMES - IN_ZONE_FRAMES

sample_csv = demo[
    ["동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"]
].to_csv(index=False).encode("utf-8-sig")


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
    out["제안 방법 DANGER 비율"] = (
        out["제안 방법 DANGER"] / out["구역 내 프레임"] * 100
    )
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

    out["Baseline DANGER 비율"] = (
        out["Baseline DANGER"] / out["구역 내 프레임"] * 100
    )
    out["제안 방법 DANGER 비율"] = (
        out["제안 방법 DANGER"] / out["구역 내 프레임"] * 100
    )
    return out


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("분석 데이터")
    mode = st.radio(
        "데이터 선택",
        ["예제 데이터", "CSV 업로드"],
        index=0,
        help="예제 데이터로 기능을 확인하거나, 직접 CSV를 업로드할 수 있습니다.",
    )

    uploaded_file = None
    if mode == "CSV 업로드":
        uploaded_file = st.file_uploader(
            "CSV 파일 업로드",
            type=["csv"],
            help="동작별 요약 CSV 또는 condition / in_zone / status 열이 포함된 CSV를 지원합니다.",
        )

    st.download_button(
        "예제 CSV 다운로드",
        data=sample_csv,
        file_name="sample_safety_data.csv",
        mime="text/csv",
        width="stretch",
    )

    st.divider()
    st.caption("학습용 안전 모니터링 프로토타입")
    st.caption("MediaPipe Pose 기반 관절 추정 결과 활용")


# ----------------------------
# Parse uploaded data
# ----------------------------
summary = demo.copy()
raw_df = None
using_demo = True

if mode == "CSV 업로드" and uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        parsed = build_summary_from_aggregate(raw_df)
        if parsed is None:
            parsed = build_summary_from_raw(raw_df)

        if parsed is None:
            st.sidebar.warning("CSV 열 구조를 자동으로 인식하지 못했습니다.")
            st.sidebar.caption("예제 데이터가 대신 표시됩니다.")
        else:
            summary = parsed
            using_demo = False
            st.sidebar.success("CSV 분석 완료")
    except Exception as e:
        st.sidebar.error(f"CSV 읽기 오류: {e}")


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


# ----------------------------
# Header: service identity
# ----------------------------
hero_text, hero_visual = st.columns([1.05, 0.95], vertical_alignment="center")

with hero_text:
    st.markdown('<span class="hero-kicker">AI Safety Dashboard</span>', unsafe_allow_html=True)
    st.title("SafePose AI")
    st.caption("MediaPipe Pose 기반 작업자 안전 모니터링 대시보드")
    st.markdown(
        "현장 안전 관리자가 **관절 움직임 분석 결과와 위험 상태를 빠르게 확인**할 수 있도록 "
        "구성한 학습용 AI 대시보드입니다."
    )
    st.markdown(
        '<div class="badge-row">'
        '<span class="badge">Pose Estimation</span>'
        '<span class="badge">Safety Monitoring</span>'
        '<span class="badge">CSV Analytics</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "현재 웹 버전은 실시간 카메라 추론이 아니라, MediaPipe Pose로 사전 분석한 결과 CSV를 "
        "불러와 요약·비교·시각화합니다."
    )

with hero_visual:
    st.image(
        "assets/safepose_pipeline.svg",
        alt="웹캠 영상이 MediaPipe Pose 관절 추정과 움직임 특징 분석을 거쳐 NORMAL, CAUTION, DANGER 상태로 분류되는 흐름",
        width="stretch",
    )

intro1, intro2, intro3 = st.columns(3)
with intro1:
    st.markdown(
        '<div class="feature-card"><div class="feature-label">👤 대상 사용자</div>'
        '<div class="feature-value">현장 안전 관리자</div></div>',
        unsafe_allow_html=True
    )
with intro2:
    st.markdown(
        '<div class="feature-card"><div class="feature-label">🤖 AI 활용 지점</div>'
        '<div class="feature-value">MediaPipe Pose 관절 좌표 추정</div></div>',
        unsafe_allow_html=True
    )
with intro3:
    st.markdown(
        '<div class="feature-card"><div class="feature-label">📊 현재 웹 기능</div>'
        '<div class="feature-value">CSV 분석 · 상태 요약 · 동작 비교</div></div>',
        unsafe_allow_html=True
    )

st.divider()

# ----------------------------
# KPI
# ----------------------------
st.subheader("현재 상태 요약")
mode_label = "예제 실험 데이터" if using_demo else "업로드 데이터"
st.caption(f"현재 분석 데이터: {mode_label}")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="kpi-card kpi-total"><div class="kpi-label">전체 프레임</div><div class="kpi-value">{total_frames:,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="kpi-card kpi-normal"><div class="kpi-label">NORMAL</div><div class="kpi-value">{normal_frames:,}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="kpi-card kpi-zone"><div class="kpi-label">위험구역 진입</div><div class="kpi-value">{in_zone_frames:,}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="kpi-card kpi-caution"><div class="kpi-label">CAUTION</div><div class="kpi-value">{caution_frames:,}</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="kpi-card kpi-danger"><div class="kpi-label">DANGER</div><div class="kpi-value">{danger_frames:,}</div></div>', unsafe_allow_html=True)

# ----------------------------
# Main dashboard tabs
# ----------------------------
tab_dashboard, tab_data, tab_service = st.tabs(
    ["대시보드", "CSV 분석", "AI 분석 구조"]
)

with tab_dashboard:
    st.subheader("위험 상태 시각화")
    st.markdown(
        '<div class="section-note">상태별 분포와 동작별 DANGER 비율을 함께 확인해 '
        '현재 분석 결과의 특징을 빠르게 파악할 수 있습니다.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([0.9, 1.1])

    with left:
        state_df = pd.DataFrame({
            "상태": ["NORMAL", "CAUTION", "DANGER"],
            "프레임": [normal_frames, caution_frames, danger_frames],
        })
        donut = px.pie(
            state_df,
            names="상태",
            values="프레임",
            hole=0.62,
            color="상태",
            color_discrete_map={
                "NORMAL": "#22A06B",
                "CAUTION": "#E2A400",
                "DANGER": "#E5484D",
            },
        )
        donut.update_traces(textinfo="percent+label")
        donut.update_layout(
            title="전체 상태 분포",
            margin=dict(t=55, b=15, l=10, r=10),
            legend_title_text="상태",
        )
        st.plotly_chart(donut, width="stretch")

    with right:
        bar = summary[["동작", "제안 방법 DANGER 비율"]].copy()
        danger_fig = px.bar(
            bar,
            x="동작",
            y="제안 방법 DANGER 비율",
            text=bar["제안 방법 DANGER 비율"].map(lambda x: f"{x:.1f}%"),
            labels={"제안 방법 DANGER 비율": "DANGER 비율 (%)"},
        )
        danger_fig.update_traces(
            textposition="outside",
            marker_color="#E5484D",
        )
        danger_fig.update_layout(
            title="동작별 DANGER 비율",
            yaxis_range=[
                0,
                max(12, float(bar["제안 방법 DANGER 비율"].max()) * 1.35),
            ],
            showlegend=False,
            margin=dict(t=55, b=15, l=10, r=10),
        )
        st.plotly_chart(danger_fig, width="stretch")

    st.subheader("Baseline 대비 제안 방법")
    compare = summary[
        ["동작", "Baseline DANGER 비율", "제안 방법 DANGER 비율"]
    ].melt(
        id_vars="동작",
        var_name="방법",
        value_name="DANGER 비율",
    )
    compare["방법"] = compare["방법"].replace({
        "Baseline DANGER 비율": "Baseline",
        "제안 방법 DANGER 비율": "제안 방법",
    })

    compare_fig = px.bar(
        compare,
        x="동작",
        y="DANGER 비율",
        color="방법",
        barmode="group",
        text=compare["DANGER 비율"].map(lambda x: f"{x:.1f}%"),
        labels={"DANGER 비율": "DANGER 비율 (%)"},
        color_discrete_map={
            "Baseline": "#9DBFF5",
            "제안 방법": "#275DAD",
        },
    )
    compare_fig.update_traces(textposition="outside")
    compare_fig.update_layout(
        yaxis_range=[0, 112],
        legend_title_text="방법",
        margin=dict(t=20, b=15, l=10, r=10),
    )
    st.plotly_chart(compare_fig, width="stretch")

    st.subheader("핵심 해석")
    max_row = summary.loc[summary["제안 방법 DANGER 비율"].idxmax()]

    with st.container(border=True):
        st.markdown(
            f"<b>{max_row['동작']}</b>에서 가장 높은 DANGER 비율 "
            f"<b>{max_row['제안 방법 DANGER 비율']:.1f}%</b>가 나타났습니다.",
            unsafe_allow_html=True
        )

        if using_demo:
            st.markdown(
                "• 정상 통과 <b>B(2.7%)</b>와 급정지 <b>C(3.3%)</b>의 차이는 "
                "<b>0.6%p</b>로 작아, 현재 규칙만으로 급정지 동작을 명확히 구분하는 데 한계가 있습니다.",
                unsafe_allow_html=True
            )
            st.markdown(
                "• <b>D 팔꿈치 움직임(8.6%)</b>은 B와 C보다 높은 DANGER 비율을 보였습니다.",
                unsafe_allow_html=True
            )

        st.caption(
            "이 수치는 모의 실험의 프레임 단위 상태 분포이며, "
            "실제 사고 발생 가능성 또는 사고 검출 정확도를 의미하지 않습니다."
        )

with tab_data:
    st.subheader("동작별 분석 결과")

    display = summary[
        [
            "동작",
            "구역 내 프레임",
            "Baseline DANGER",
            "제안 방법 DANGER",
            "제안 방법 DANGER 비율",
        ]
    ].copy()

    display["제안 방법 DANGER 비율"] = (
        display["제안 방법 DANGER 비율"] / 100
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "제안 방법 DANGER 비율": st.column_config.ProgressColumn(
                "DANGER 비율",
                help="위험구역 내부 프레임 중 DANGER 판정 비율",
                format="percent",
                min_value=0,
                max_value=1,
            )
        },
    )

    if mode == "CSV 업로드" and uploaded_file is None:
        st.info("왼쪽 사이드바에서 CSV 파일을 업로드하면 해당 데이터로 분석합니다.")

    if raw_df is not None and not using_demo:
        with st.expander("업로드 원본 데이터 미리보기"):
            st.dataframe(raw_df.head(30), hide_index=True, width="stretch")

    with st.expander("지원하는 CSV 형식"):
        st.markdown(
            """
            **동작별 요약 형식**
            - 동작
            - 구역 내 프레임
            - Baseline DANGER
            - 제안 방법 DANGER

            또는 **프레임 단위 형식**
            - condition
            - in_zone
            - status
            """
        )

with tab_service:
    st.subheader("AI 분석 구조 및 서비스 범위")

    st.markdown(
        """
        **SafePose AI**는 작업자의 관절 움직임 분석 결과를 현장 안전 관리자가
        빠르게 확인할 수 있도록 구성한 대시보드형 서비스입니다.

        **AI가 사용되는 부분**
        1. 웹캠 영상에서 **MediaPipe Pose**로 어깨·팔꿈치·손목 관절 좌표를 추정합니다.
        2. 위험구역 진입 여부와 손목 이동속도, 팔꿈치 각도 변화를 계산합니다.
        3. 규칙 기반 위험점수를 이용해 NORMAL / CAUTION / DANGER 상태로 구분합니다.
        4. Streamlit에서 결과를 요약하고 비교·시각화합니다.

        **현재 웹 버전에서 제공하는 기능**
        - 예제 실험 데이터 즉시 확인
        - 분석 결과 CSV 업로드
        - 위험 상태 KPI 및 상태 분포 시각화
        - Baseline과 제안 방법 비교
        - 동작별 결과 표와 자동 핵심 해석

        **현재 프로토타입의 범위**
        - 참가자 1명의 모의 동작 데이터를 사용했습니다.
        - A~D 동작을 각각 4회 수행한 실험 결과를 기반으로 합니다.
        - 저장된 CSV에는 반복별 시작·종료 구분이 없어 프레임 단위 비율로 비교합니다.
        - 현재 배포 버전은 **실시간 웹캠 분석 기능을 포함하지 않습니다.**
        - 실제 산업현장 적용 전에는 다양한 작업자와 환경에서 추가 검증이 필요합니다.
        """
    )

    with st.container(border=True):
        st.markdown("**서비스 이용 흐름**")
        st.write("예제 데이터 확인 → CSV 업로드 → 상태 요약 → 동작별 비교 → 결과 해석")
