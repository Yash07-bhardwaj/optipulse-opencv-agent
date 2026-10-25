from opencv_tools import diagnose_frame, adaptive_remediation, run_structural_inspection

def run_agentic_vision_loop(img):
    m1 = diagnose_frame(img)
    i1 = run_structural_inspection(img, 70, 180)
    plan = {"suppress_glare": m1["glare_pct"] > 1.5,
            "apply_clahe": m1["brightness"] < 95 or m1["contrast"] < 42,
            "clahe_clip": 3.5 if m1["brightness"] < 65 else 2.5,
            "sharpen": m1["blur_score"] < 140.0,
            "adjusted_canny": (35, 110) if m1["blur_score"] < 140.0 else (50, 150)}
    enh, ops, lat = adaptive_remediation(img, plan)
    m2 = diagnose_frame(enh)
    i2 = run_structural_inspection(enh, *plan["adjusted_canny"])
    delta = round(i2["confidence_pct"] - i1["confidence_pct"], 1)
    dec = "AUTONOMOUS_APPROVAL: Logged to AWS pipeline." if i2["confidence_pct"] >= 75 else "HUMAN_ESCALATION"
    trace = [
        {"step": 1, "phase": "PERCEPTION", "metrics": m1, "confidence": i1["confidence_pct"]},
        {"step": 2, "phase": "DECISION (Policy Synthesis)", "updated_plan": plan},
        {"step": 3, "phase": "ACTION (OpenCV 5 Execution)", "ops": ops, "post_metrics": m2, "latency_ms": lat},
        {"step": 4, "phase": "GOVERNANCE", "decision": dec, "confidence_delta": f"+{delta}%"}
    ]
    return {"baseline_image_b64": i1["annotated_b64"], "enhanced_image_b64": i2["annotated_b64"],
            "baseline_metrics": m1, "post_metrics": m2, "baseline_inspection": i1,
            "final_inspection": i2, "trace_log": trace, "final_decision": dec}
