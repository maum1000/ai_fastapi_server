# main.py
import face_recognition
import logging
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import logging  # 로그 출력을 위한 모듈
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger('pybo')  # 'pybo'라는 로거 생성

# 각 단계별 클래스를 개별적으로 임포트합니다.
from ai_system import Pipeline, Data, BaseConfig, steps, factories

def process_image(image_path, selected_detectors):
    """
    이미지를 처리하는 메인 함수입니다. 얼굴을 탐지하고, 인코딩(임베딩)한 결과를 반환합니다.

    Parameters
    ----------
    image_path : str
        처리할 이미지의 경로입니다.
    selected_detectors : list
        사용할 얼굴 탐지기(detectors)의 목록입니다.

    Returns
    -------
    output_image_path : str
        처리된 이미지가 저장된 경로입니다.
    encodings : list
        얼굴의 인코딩(숫자로 변환된 얼굴 특징 값)을 반환합니다.
    """
    # 설정 정보를 가져옵니다.
    config = BaseConfig.get_config()

    # 탐지기를 선택된 종류에 맞게 생성합니다.
    detectors = []
    for detector in selected_detectors:
        if detector == 'mtcnn':  # 'mtcnn' 탐지기는 사용하지 않음
            continue
        # 탐지기 생성하여 리스트에 추가
        detectors.append(factories.FaceDetectorFactory.create(detector, config[f'{detector}']))

    # 파이프라인 설정: 얼굴 탐지와 인코딩을 수행하는 단계 추가
    pipeline = Pipeline()
    pipeline.add(steps.FaceDetector(detectors))  # 얼굴 탐지 단계 추가
    pipeline.add(steps.FaceEncoder())  # 얼굴 인코딩 단계 추가

    logging.info(f"이미지 처리 시작: {image_path}")

    # 데이터 객체를 생성하고 이미지 경로를 설정합니다.
    data = Data(config, image_path)
    data.image_rgb  # 이미지의 RGB 값 설정

    # 파이프라인 실행: 설정된 단계를 순차적으로 수행
    pipeline.run(data)

    logging.info(f"이미지 처리 완료: {image_path}")

    # 처리된 이미지의 경로와 얼굴 인코딩 반환
    output_image_path = data.output_image_path
    encodings = data.encodings
    return output_image_path, encodings

def compare_faces(image1_path: str, image2_path: str, selected_detectors: list = ['yolo']) -> float:
    """
    두 이미지의 얼굴을 비교하여 유사도를 계산합니다.

    Parameters
    ----------
    image1_path : str
        첫 번째 이미지의 경로입니다.
    image2_path : str
        두 번째 이미지의 경로입니다.
    selected_detectors : list, optional
        사용할 얼굴 탐지기의 목록 (기본값은 ['yolo']).

    Returns
    -------
    similarity : float
        두 얼굴 간의 유사도를 나타내는 값입니다. (0에 가까울수록 유사함)
    """
    logger.info(f"이미지 1 인코딩 중: {image1_path}")
    # 첫 번째 이미지 처리
    _, face_encoding1 = process_image(image1_path, selected_detectors)

    logger.info(f"이미지 2 인코딩 중: {image2_path}")
    # 두 번째 이미지 처리
    _, face_encoding2 = process_image(image2_path, selected_detectors)

    # 얼굴이 정확히 1개씩 있어야 비교 가능
    if len(face_encoding1) != 1:
        raise ValueError("첫 번째 사진에 얼굴이 1개가 아닙니다.")
    elif len(face_encoding2) != 1:
        raise ValueError("두 번째 사진에 얼굴이 1개가 아닙니다.")

    # 3차원 배열을 2차원 배열로 변환 (필요한 경우)
    face_encoding1 = np.array(face_encoding1[0]).reshape(1, -1)
    face_encoding2 = np.array(face_encoding2[0]).reshape(1, -1)

    logger.info(f"얼굴 유사도 계산 중: {image1_path}, {image2_path}")

    # 코사인 유사도 계산
    similarity_score = cosine_similarity(face_encoding1, face_encoding2)

    # 유사도 스코어를 %로 변환 (0 ~ 1 -> 0% ~ 100%)
    similarity_percentage = similarity_score[0][0] * 100

    return similarity_percentage

class DetectionConfig(BaseConfig):
    # 별도로 추가할 커스터마이징이 없으면 그대로 사용
    yolo_path = 'yolov8_l_trump.pt'
    django_dir = BASE_DIR
    results_folder = os.path.join(django_dir, 'media', 'detection', 'a_image1')

def detect_president(image_path: str, selected_detectors: list = ['yolo']) -> str:
    """
    이미지를 처리하고, 얼굴 탐지 결과 바운딩 박스를 이미지에 그려 저장하는 함수입니다.

    Parameters
    ----------
    image_path : str
        처리할 이미지의 경로입니다.
    selected_detectors : list, optional
        사용할 얼굴 탐지기의 목록 (기본값은 ['yolo']).

    Returns
    -------
    output_image_path : str
        처리된 이미지가 저장된 경로입니다.
    """
    # 설정 정보를 가져옵니다.
    config = DetectionConfig.get_config()

    # 탐지기를 선택된 종류에 맞게 생성합니다.
    detectors = []
    for detector in selected_detectors:
        if detector == 'mtcnn':  # 'mtcnn' 탐지기는 사용하지 않음
            continue
        detectors.append(factories.FaceDetectorFactory.create(detector, config[f'{detector}']))

    # 파이프라인 설정: 얼굴 탐지, 정보 그리기, 저장하는 단계 추가
    pipeline = Pipeline()
    pipeline.add(steps.FaceDetector(detectors))  # 얼굴 탐지 단계 추가
    pipeline.add(steps.InfoDrawer(thickness=5))  # 탐지 정보 그리기 단계 추가
    pipeline.add(steps.Saver())  # 이미지 저장 단계 추가

    logging.info(f"이미지 처리 시작: {image_path}")

    # 데이터 객체를 생성하고 이미지 경로를 설정합니다.
    data = Data(config, image_path)
    data.image_rgb  # 이미지의 RGB 값 설정

    # 파이프라인 실행: 설정된 단계를 순차적으로 수행
    pipeline.run(data)

    logging.info(f"이미지 처리 완료: {image_path}")

    # 처리된 이미지의 경로 반환
    output_image_path = data.output_image_path

    # 서버에서 템플릿 렌더링시 사용할 상대 경로 반환
    delete_path = os.path.join(BASE_DIR, 'media')
    return output_image_path