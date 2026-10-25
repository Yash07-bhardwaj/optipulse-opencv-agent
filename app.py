import streamlit as st
import cv2, numpy as np, base64, time, platform
from agent_engine import run_agentic_vision_loop
from opencv_tools import diagnose_frame, adaptive_remediation, run_structural_inspection

st.set_page_config(page_title="OptiPulse | OpenCV 5 & AWS COOL", layout="wide")
st.title("OptiPulse: Self-Tuning Agentic Vision on AWS Graviton (COOL)")
st.caption(f"OpenCV AI Competition 2026 | Runtime: OpenCV {cv2.__version__} ({platform.machine()})")

def make_pcb():
    img = np.full((480, 640, 3), 35, dtype=np.uint8)
    for i in range(60, 600, 80): cv2.line(img, (i, 40), (i, 440), (60, 140, 60), 3)
    for j in range(80, 420, 70): cv2.line(img, (40, j), (600, j), (60, 140, 60), 2)
    cv2.rectangle(img, (150, 120), (270, 220), (90, 90, 90), -1)
    cv2.circle(img, (210, 170), 22, (40, 40, 190), -1)
    img = cv2.GaussianBlur((img * 0.55).astype(np.uint8), (9, 9), 2.8)
    cv2.circle(img, (310, 200), 45, (252, 252, 252), -1)
    return img

t1, t2 = st.tabs(["Active Perception Agent Loop", "AWS Graviton (COOL) Benchmark"])
with t1:
    img = make_pcb()
    st.image(img, channels="BGR", caption="Input Degraded Frame (Glare + Blur + Low Contrast)", width=420)
    if st.button("Execute Multi-Step Agentic Vision Loop", type="primary"):
        res = run_agentic_vision_loop(img)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pass 1: Baseline OpenCV 5 (Degraded)**")
            st.image(base64.b64decode(res["baseline_image_b64"]), channels="BGR")
            st.metric("Baseline Confidence", str(res["baseline_inspection"]["confidence_pct"]) + "%")
            st.json(res["baseline_metrics"])
        with c2:
            st.markdown("**Pass 2: Post-Agentic Remediation (Tuned OpenCV 5)**")
            st.image(base64.b64decode(res["enhanced_image_b64"]), channels="BGR")
            st.metric("Post-Remediation Confidence", str(res["final_inspection"]["confidence_pct"]) + "%")
            st.json(res["post_metrics"])
        st.success(res["final_decision"])
        st.json(res["trace_log"])

with t2:
    st.markdown(f"**Architecture:** `{platform.machine()}` | **OpenCV:** `{cv2.__version__}`")
    if st.button("Run Reproducible Workload Benchmark (50 Frames)"):
        s = make_pcb()
        lats = []
        for _ in range(50):
            t0 = time.perf_counter()
            enh, _, _ = adaptive_remediation(s, {"suppress_glare": True, "apply_clahe": True, "sharpen": True})
            run_structural_inspection(enh)
            lats.append((time.perf_counter() - t0)*1000)
        m1, m2, m3 = st.columns(3)
        m1.metric("Throughput", f"{1000/np.mean(lats):.1f} FPS")
        m2.metric("Mean Latency", f"{np.mean(lats):.2f} ms")
        m3.metric("p95 Latency", f"{np.percentile(lats, 95):.2f} ms")
