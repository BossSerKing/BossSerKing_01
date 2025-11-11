import streamlit as st
import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
import pandas as pd
import os

# ====== เตรียมโฟลเดอร์สำหรับเก็บภาพแจ้งเตือน ======
os.makedirs("alerts", exist_ok=True)

# ====== Class อันตราย ======
DANGER_LABELS = {"gun", "knife", "pistol", "riffle", "drug"}

# ====== เก็บ log การตรวจจับ ======
detection_log = []

# ====== คำนวณระดับความเสี่ยง ======
def calculate_risk(danger_items):
    score = 0
    for label, count in danger_items.items():
        if label in {"gun", "pistol", "riffle"}:
            score += 3 * count
        elif label == "knife":
            score += 2 * count
        elif label == "drug":
            score += 1 * count
    
    if score >= 5:
        return "🔴 HIGH RISK", "red"
    elif score >= 2:
        return "🟠 MEDIUM RISK", "orange"
    elif score > 0:
        return "🟡 LOW RISK", "yellow"
    return "🟢 SAFE", "green"

# ====== โหลดโมเดล (Cache ให้โหลดครั้งเดียว) ======
@st.cache_resource
def load_model():
    return YOLO("best_final.pt")

# ====== ฟังก์ชันสตรีมกล้อง ======
def video_stream(model, confidence_threshold, resolution):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ใช้ CAP_DSHOW ป้องกันปัญหาเว็บแคมบน Windows
    cap.set(cv2.CAP_PROP_FPS, 60)
    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        st.error("Cannot access the webcam. Please check your camera connection.")
        return

    frame_placeholder = st.empty()
    label_count_placeholder = st.empty()
    danger_count_placeholder = st.empty()
    risk_placeholder = st.empty()
    fps_placeholder = st.empty()

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot read frame from webcam.")
            break

        # คำนวณ FPS
        current_time = time.time()
        frame_count += 1
        elapsed_time = current_time - prev_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0

        # รัน YOLO
        results = model(frame, conf=confidence_threshold)
        label_counts = defaultdict(int)
        danger_items = {}

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                if conf >= confidence_threshold:
                    label = model.names[int(cls)]
                    label_counts[label] += 1

                    # วาดกรอบ
                    color = (0, 0, 255) if label in DANGER_LABELS else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # เก็บ log
                    detection_log.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "label": label,
                        "confidence": float(conf),
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
                    })

                    # ถ้าเจออันตราย -> ครอปภาพเก็บไว้
                    if label in DANGER_LABELS:
                        danger_items[label] = label_counts[label]
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        alert_path = f"alerts/{label}_{int(time.time())}.jpg"
                        cv2.imwrite(alert_path, crop)

        # แปลงภาพเป็น RGB และแสดงในเว็บ
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # แสดงจำนวนวัตถุทั้งหมด
        label_count_placeholder.markdown("### ✅ Object Counts:")
        for label, count in label_counts.items():
            label_count_placeholder.write(f"- **{label}**: {count}")

        # แสดงจำนวนวัตถุอันตราย
        if danger_items:
            danger_count_placeholder.markdown("### 🔴 Danger Detected!")
            for label, count in danger_items.items():
                danger_count_placeholder.write(f"- 🚨 **{label}**: {count}")
        else:
            danger_count_placeholder.empty()

        # แสดง Risk Level
        risk_level, color = calculate_risk(danger_items)
        risk_placeholder.markdown(f"## <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)

        # แสดง FPS
        fps_placeholder.markdown(f"### 🎥 FPS: `{fps:.2f}`")

        # ให้ UI มีเวลาทำงาน
        time.sleep(0.01)

    cap.release()
    # บันทึก Log ลง CSV
    if detection_log:
        df = pd.DataFrame(detection_log)
        df.to_csv("detection_log.csv", index=False)
        st.success("📝 Detection log saved as `detection_log.csv`")

# ====== Main ======
def main():
    st.title("🔍 Real-Time Security Object Detection (YOLOv8)")
    st.markdown("ระบบตรวจจับวัตถุเพื่อความปลอดภัยแบบเรียลไทม์ พร้อมประเมินความเสี่ยง")

    # โหลดโมเดล
    model = load_model()

    # ค่าตั้งต้นของปุ่ม Start/Stop
    if "running" not in st.session_state:
        st.session_state.running = False

    # ตั้งค่า Confidence
    confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

    # ตั้งค่าความละเอียด
    resolution_option = st.selectbox(
        "📷 Select Camera Resolution",
        ["1280x720", "640x480", "1920x1080"],
        index=0
    )
    resolution_map = {
        "1920x1080": (1920, 1080),
        "1280x720": (1280, 720),
        "640x480": (640, 480),
    }
    resolution = resolution_map[resolution_option]

    # ปุ่มควบคุมกล้อง
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Webcam"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop Webcam"):
            st.session_state.running = False

    # เริ่มทำงานถ้ากด Start
    if st.session_state.running:
        video_stream(model, confidence_threshold, resolution)

main()