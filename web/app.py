import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 쓰레기 분류기", layout="centered")

# --- 2. 모델 로드 (캐싱 처리로 속도 향상) ---
@st.cache_resource
def load_yolo_model():
    # 파일명이 다르면 여기서 수정하세요 (예: 'best.pt')
    model = YOLO("best.pt") 
    return model

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다. 'best.pt' 파일이 같은 폴더에 있는지 확인하세요. 에러: {e}")
    st.stop()

# --- 3. UI 부분 ---
st.title("♻️ 스마트 쓰레기 분리배출 도우미")
st.write("사진을 올리면 AI가 어떤 쓰레기인지 분석하고 분리배출 방법을 알려드립니다.")

uploaded_file = st.file_uploader("분석할 쓰레기 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

# --- 4. 메인 로직 ---
if uploaded_file is not None:
    # 이미지 열기
    image = Image.open(uploaded_file)
    
    # 두 개의 칼럼으로 나누어 보기 좋게 배치
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="원본 이미지", use_container_width=True)
    
    # 분석 시작
    with st.spinner("AI 분석 중..."):
        results = model(image)
        
    with col2:
        # 결과 이미지 그리기
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="분석 결과", use_container_width=True)

    # --- 5. 상세 탐지 결과 출력 ---
    st.divider()
    st.subheader("🔍 탐지된 물체 분석 결과")
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        # 결과 데이터를 테이블 형태로 보여주기 위한 리스트
        detection_data = []
        
        for box in boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detection_data.append({
                "물체 종류": label,
                "확률(Confidence)": f"{confidence:.1%}"
            })
            
            # 텍스트로 한 번 더 출력
            st.write(f"📍 **{label}** 이(가) 발견되었습니다. (확률: {confidence:.1%})")
        
        # 표 형식으로 정리해서 보여주기
        st.table(detection_data)
        st.success(f"총 {len(boxes)}개의 물체를 성공적으로 분류했습니다.")
    else:
        st.warning("탐지된 물체가 없습니다. 사진을 다시 찍어보세요.")

# --- 6. 추가 안내 (바닥글) ---
st.info("💡 팁: 밝은 곳에서 물체가 잘 보이게 촬영하면 정확도가 올라갑니다.")