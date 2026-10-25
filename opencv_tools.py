import cv2, numpy as np, time, base64

def encode_image(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")

def diagnose_frame(img):
    t0 = time.perf_counter()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    _, m = cv2.threshold(g, 245, 255, cv2.THRESH_BINARY)
    glare = float(np.sum(m > 0) / g.size) * 100.0
    return {"blur_score": round(lap, 2), "brightness": round(float(np.mean(g)), 2),
            "contrast": round(float(np.std(g)), 2), "glare_pct": round(glare, 2),
            "latency_ms": round((time.perf_counter() - t0)*1000, 2)}

def adaptive_remediation(img, plan):
    t0 = time.perf_counter()
    p, ops = img.copy(), []
    if plan.get("suppress_glare"):
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(g, 240, 255, cv2.THRESH_BINARY)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        p = cv2.inpaint(p, m, 3, cv2.INPAINT_TELEA)
        ops.append("cv2.inpaint (Telea Specular Glare Removal)")
    if plan.get("apply_clahe"):
        c = plan.get("clahe_clip", 3.0)
        l, a, b = cv2.split(cv2.cvtColor(p, cv2.COLOR_BGR2LAB))
        cl = cv2.createCLAHE(clipLimit=c, tileGridSize=(8, 8)).apply(l)
        p = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        ops.append(f"cv2.createCLAHE (LAB Space, clip={c})")
    if plan.get("sharpen"):
        d = cv2.bilateralFilter(p, 7, 50, 50)
        p = cv2.addWeighted(d, 1.6, cv2.GaussianBlur(d, (0, 0), 2.0), -0.6, 0)
        ops.append("cv2.bilateralFilter + Unsharp Masking")
    return p, ops, round((time.perf_counter() - t0)*1000, 2)

def run_structural_inspection(img, canny_low=50, canny_high=150):
    t0 = time.perf_counter()
    ann, g = img.copy(), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), canny_low, canny_high)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 250:
            x, y, w, h = cv2.boundingRect(c)
            rois.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": round(area, 1)})
            cv2.rectangle(ann, (x, y), (x+w, y+h), (0, 255, 120), 2)
    kps, _ = cv2.ORB_create(nfeatures=500).detectAndCompute(g, None)
    ann = cv2.drawKeypoints(ann, kps, None, color=(255, 140, 0), flags=0)
    conf = min(98.5, round(40.0 + (len(kps)/500.0)*45.0 + min(len(rois)*2.5, 15.0), 1))
    return {"anomaly_count": len(rois), "orb_keypoints": len(kps), "confidence_pct": conf,
            "rois": rois[:8], "latency_ms": round((time.perf_counter() - t0)*1000, 2),
            "annotated_b64": encode_image(ann)}
