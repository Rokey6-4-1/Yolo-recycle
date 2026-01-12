# from ultralytics import YOLO





# def main():
#     # 1. 모델 체급 상향 (Medium 모델)
#     model = YOLO('Recycle_Final_Project/YOLO11m_HighRes/weights/last.pt') # 반드시 last.pt 사용
#     model.train(resume=True)
#     print("\n" + "="*50)
#     print("재활용 분류 모델 학습 시작")
#     print(f" 클래스 수: 12개 / 학습 이미지: 15,000장 / 목표 시간: 90분")
#     print("="*50)
#     # 2. 고성능 단거리 질주 학습 (50 에포크)
#     model.train(
#         data='data_final.yaml',      # 기존 정제 데이터 사용
#         epochs=30,                   # 시간 관계상 30회 집중 학습
#         imgsz=640,                   # 시력(해상도) 상향은 포기할 수 없는 핵심!
#         batch=8,                     # VRAM 부하 방지용
#         patience=10,                 # 10회 연속 개선 없으면 조기 종료
#         device='0',                  # GPU 4060 가동
#         workers=4,                   # CPU 발열 관리 모드
#         project='Recycle_Final_Project',
#         name='YOLO11m_HighRes', # 최종 버전 이름
#         exist_ok=True,
#         optimizer='AdamW',
#         seed=0,
#         close_mosaic=5              # 마지막 10회는 정밀 사격 모드 유지
#     )


# if __name__ == '__main__':
#     main()


from ultralytics import YOLO

def main():
    # 1. 멈췄던 시점의 모델 불러오기 (경로는 본인의 환경에 맞게 확인)
    # 22 에포크 진행 중 멈췄으므로 last.pt에 모든 기록이 남아있습니다.
    model = YOLO('Recycle_Final_Project/YOLO11m_HighRes/weights/last.pt')

    print("\n" + "="*50)
    print("중단된 학습을 재개합니다 (Resume Mode)")
    print("이전 설정값(30 에포크, 640 해상도 등)이 그대로 유지됩니다.")
    print("="*50)

    # 2. 재개(Resume) 실행
    # 이 한 줄이면 이전에 설정했던 30 에포크, batch, workers 등이 자동으로 로드됩니다.
    model.train(resume=True)

if __name__ == '__main__':
    main()