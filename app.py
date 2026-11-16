import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI 작업자 안전 모니터링",
    layout="wide"
)

st.title("AI 작업자 안전 모니터링")
st.caption("MediaPipe Pose 기반 관절 움직임 분석 대시보드")

st.info(
    "본 대시보드는 MediaPipe Pose로 추정한 관절 움직임과 위험구역 진입 정보를 "
    "바탕으로 NORMAL, CAUTION, DANGER 상태를 시각화하는 학습용 프로토타입입니다. "
    "실제 산업재해 발생 가능성을 예측하는 시스템은 아닙니다."
)

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
            "현재 버전은 동작별 요약 CSV 또는 "
            "condition / in_zone / status 열이 포함된 프레임 단위 CSV를 지원합니다."
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
        zone_bool = zone_values.astype(float) > 0

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
    required_sets = [
        {"동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"},
        {"condition", "in_zone_frames", "baseline_danger", "proposed_danger"},
    ]

    if required_sets[0].issubset(df.columns):
        out = df[list(required_sets[0])].copy()
        out = out[["동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"]]
    elif required_sets[1].issubset(df.columns):
        out = df[["condition", "in_zone_frames", "baseline_danger", "proposed_danger"]].copy()
        out.columns = ["동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"]
        out["동작"] = out["동작"].apply(normalize_condition_name)
    else:
        return None

    for c in ["구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna()
    out["Baseline DANGER 비율"] = (
        out["Baseline DANGER"] / out["구역 내 프레임"] * 100
    )
    out["제안 방법 DANGER 비율"] = (
        out["제안 방법 DANGER"] / out["구역 내 프레임"] * 100
    )
    return out

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
            st.warning(
                "CSV 열 구조를 자동으로 인식하지 못해 예제 데이터를 표시합니다. "
                "아래에서 업로드한 CSV의 열 이름을 확인해 주세요."
            )
            st.write("업로드한 CSV 열:", list(raw_df.columns))
        else:
            summary = parsed
            using_demo = False
            st.success("업로드한 CSV를 분석했습니다.")
    except Exception as e:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")

# ----------------------------
# KPI
# ----------------------------
st.subheader("위험 상태 요약")

if using_demo:
    total_frames = TOTAL_FRAMES
    in_zone_frames = IN_ZONE_FRAMES
    danger_frames = DANGER_FRAMES
    caution_frames = CAUTION_FRAMES
else:
    in_zone_frames = int(summary["구역 내 프레임"].sum())
    danger_frames = int(summary["제안 방법 DANGER"].sum())
    caution_frames = max(in_zone_frames - danger_frames, 0)
    total_frames = len(raw_df) if raw_df is not None else in_zone_frames

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 프레임", f"{total_frames:,}")
c2.metric("위험구역 진입", f"{in_zone_frames:,}")
c3.metric("CAUTION", f"{caution_frames:,}")
c4.metric("DANGER", f"{danger_frames:,}")

st.divider()

# ----------------------------
# Charts
# ----------------------------
left, right = st.columns(2)

with left:
    st.subheader("동작별 DANGER 비율")
    bar = summary[["동작", "제안 방법 DANGER 비율"]].copy()
    fig = px.bar(
        bar,
        x="동작",
        y="제안 방법 DANGER 비율",
        text=bar["제안 방법 DANGER 비율"].map(lambda x: f"{x:.1f}%"),
        labels={"제안 방법 DANGER 비율": "DANGER 비율 (%)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, max(12, float(bar["제안 방법 DANGER 비율"].max()) * 1.35)])
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Baseline 대비 제안 방법")
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
        labels={"DANGER 비율": "DANGER 비율 (%)"},
    )
    fig2.update_layout(yaxis_range=[0, 110])
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Table
# ----------------------------
st.subheader("동작별 분석 결과")

display = summary[[
    "동작", "구역 내 프레임", "Baseline DANGER", "제안 방법 DANGER",
    "제안 방법 DANGER 비율"
]].copy()
display["제안 방법 DANGER 비율"] = display["제안 방법 DANGER 비율"].map(
    lambda x: f"{x:.1f}%"
)
st.dataframe(display, use_container_width=True, hide_index=True)

# ----------------------------
# Interpretation
# ----------------------------
st.subheader("결과 해석")

if len(summary) > 0:
    max_row = summary.loc[summary["제안 방법 DANGER 비율"].idxmax()]
    st.write(
        f"- 현재 데이터에서는 **{max_row['동작']}**에서 "
        f"가장 높은 DANGER 비율 **{max_row['제안 방법 DANGER 비율']:.1f}%**가 나타났습니다."
    )

if using_demo:
    st.write(
        "- 정상 통과 B(2.7%)와 급정지 C(3.3%)의 차이는 0.6%p로 작아, "
        "현재 규칙만으로 급정지 동작을 명확하게 구분하는 데 한계가 있습니다."
    )
    st.write(
        "- D 팔꿈치 움직임은 8.6%로 B와 C보다 높은 DANGER 비율을 보였습니다."
    )

st.warning(
    "주의: 본 결과는 모의 실험의 프레임 단위 상태 분포이며, "
    "실제 끼임 사고의 발생 가능성이나 사고 검출 정확도를 의미하지 않습니다."
)

with st.expander("서비스 및 분석 방법"):
    st.markdown(
        """
        **분석 흐름**
        1. 웹캠 영상에서 MediaPipe Pose로 어깨·팔꿈치·손목 관절 좌표를 추정합니다.
        2. 위험구역 진입 여부와 손목 이동속도, 팔꿈치 각도 변화를 계산합니다.
        3. 규칙 기반 위험점수를 이용해 NORMAL / CAUTION / DANGER로 구분합니다.
        4. Streamlit 대시보드에서 동작별 결과를 시각화합니다.

        **현재 버전의 범위**
        - 1명의 참가자가 수행한 모의 동작 기반의 학습용 프로토타입입니다.
        - 실제 산업현장 적용 전에는 다양한 작업자와 환경에서 추가 검증이 필요합니다.
        """
    )
