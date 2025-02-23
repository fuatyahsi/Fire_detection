import cv2
import streamlit as st
import tempfile
import os
import requests
import time
import torch
from ultralytics import YOLO
from PIL import Image

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

# Pushbullet API ile bildirim gönderme fonksiyonu
def send_push_notification(message):
    pushbullet_api_key = "YOUR_PUSHBULLET_API_KEY"  # API anahtarını buraya ekle
    url = "https://api.pushbullet.com/v2/pushes"
    headers = {"Access-Token": pushbullet_api_key}
    data = {"type": "note", "title": "Yangın veya Duman Tespiti", "body": message}
    requests.post(url, headers=headers, data=data)

# Tahmin fonksiyonu: Modelin sonuçlarını kullanarak kutuları ve Türkçe etiketleri manuel çizeriz.
def predict_frame(frame, conf_threshold, iou_threshold, image_size):
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=image_size
    )
    # Sonuçların tek elemanlı olduğunu varsayıyoruz
    r = results[0]
    
    # frame, OpenCV'den okunduğundan BGR formatındadır. Üzerinde çizim yapalım.
    im_array = frame.copy()

    for box in r.boxes:
        # box.xyxy: [x1, y1, x2, y2]
        coords = box.xyxy.cpu().numpy().flatten()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, coords)
        
        # Modelin tahmin ettiği orijinal etiket (örneğin "Fire" veya "Smoke")
        label_en = r.names[int(box.cls)]
        # Türkçeleştirme
        if label_en.lower() == "Fire":
            label_tr = "Ateş"
        elif label_en.lower() == "smoke":
            label_tr = "Duman"
        else:
            label_tr = label_en

        # Eğer tespit başarılıysa (örn. confidence kontrolü ekleyebilirsiniz)
        # Örneğin; if box.conf > 0.25:  (opsiyonel)
        
        # Kutu çizimi
        im_array = cv2.rectangle(im_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Türkçe etiket ekleme
        im_array = cv2.putText(im_array, label_tr, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Bildirim gönderme (opsiyonel, sadece Türkçe etiketlere göre)
        if label_tr in ["Ateş", "Duman"]:
            send_push_notification("Yangın veya duman tespit edildi!")
    
    # Görseli RGB'ye dönüştür (OpenCV BGR formatında çalışır)
    im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    return im

# Video yükleme
video_file = st.file_uploader("Video Seçiniz", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Geçici bir dosyaya videoyu kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_path = temp_video_file.name

    # OpenCV ile videoyu aç
    video = cv2.VideoCapture(temp_video_path)
    
    # Streamlit'te canlı görüntü alanı oluştur
    frame_display = st.empty()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break  # Video bittiğinde döngüyü durdur

        # Frame üzerinde tahmin yap (manuel çizim kullanılıyor)
        im = predict_frame(frame, 0.25, 0.40, 640)

        # Streamlit ile tahmin sonuçlarını göster
        frame_display.image(im, caption="Developed by Fuat YAHŞİ", use_container_width=True)

        # time.sleep(0.01)  # İsteğe bağlı, kare arası bekleme

    video.release()  # Video dosyasını kapat
    os.remove(temp_video_path)  # Geçici video dosyasını sil
