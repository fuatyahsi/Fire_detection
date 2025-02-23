import cv2
import streamlit as st
import torch
import yt_dlp
import requests
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="AFAD",
    page_icon="bayrak.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={'About': "Developed by Fuat YAHŞİ"}
)

# Sayfa başlığı ve görsel
html_temp = """
<div style="background-color:red;padding:3px">
<h4 style="color:white;text-align:center;">YANGIN TESPİT MODÜLÜ</h4>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)
st.image("AFAD_page-0001.jpg", caption="Eğitim ve Farkındalık Dairesi", clamp=False)

# YOLO modeli yükle
model = YOLO("best.pt")

# Tahmin fonksiyonu
def predict_frame(frame, conf_threshold=0.25, iou_threshold=0.40, image_size=640):
    results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, imgsz=image_size)
    r = results[0]
    im_array = frame.copy()

    for box in r.boxes:
        coords = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = map(int, coords)

        label_en = r.names[int(box.cls)]
        label_tr = "Ates" if label_en.lower() == "fire" else "Duman" if label_en.lower() == "smoke" else label_en

        im_array = cv2.rectangle(im_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        im_array = cv2.putText(im_array, label_tr, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))

# YouTube videolarını işlemek için fonksiyon
def get_youtube_video_url(youtube_link):
    ydl_opts = {"format": "best[ext=mp4]", "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_link, download=False)
            return info["url"]
    except Exception as e:
        st.error(f"YouTube videosu alınırken hata oluştu: {e}")
        return None

# Genel video URL’lerini işlemek için fonksiyon
def is_valid_video_url(video_url):
    try:
        response = requests.head(video_url, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "")
        return "video" in content_type
    except Exception as e:
        st.error(f"Geçersiz video URL'si: {e}")
        return False

# Video URL üzerinden işle
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        st.error("Video açılamadı! Lütfen geçerli bir video URL'si girin.")
        return
    
    frame_display = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        im = predict_frame(frame)
        frame_display.image(im, caption="Developed by Fuat YAHŞİ", use_container_width=True)

    cap.release()

# Kullanıcıdan video kaynağı alma
st.sidebar.header("Video Kaynağı Seçimi")
video_file = st.sidebar.file_uploader("Yerel video dosyası seçiniz", type=["mp4", "mov", "avi"])
video_url = st.sidebar.text_input("YouTube veya Diğer Video URL'si")

if video_url:
    st.info("Video işleniyor, lütfen bekleyin...")
    
    if "youtube.com" in video_url or "youtu.be" in video_url:
        mp4_url = get_youtube_video_url(video_url)
        if mp4_url:
            process_video(mp4_url)
    elif is_valid_video_url(video_url):
        process_video(video_url)
    else:
        st.error("Geçersiz video URL'si. Lütfen geçerli bir video bağlantısı girin.")
    
elif video_file:
    process_video(video_file.name)
