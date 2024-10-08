# main.py

import logging
import warnings
import os
import pickle
import numpy as np

# 각 단계별 클래스를 개별적으로 임포트합니다.
from ..models.FaceDetector import FaceDetector, FaceDetectorFactory
from ..models.FacePredictor import FacePredictor, FacePredictorFactory
from ..models.FaceEncoder import FaceEncoder
from ..models.FaceMatcher import TargetFaceMatcher
from ..annotation.FaceInfoCounter import FaceInfoCounter
from ..annotation.InfoDrawer import InfoDrawer
from ..annotation.InfoWriter import InfoWriter
from ..annotation.ImageResizer import ImageResizer
from ..annotation.Saver import Saver
from ..core.config import Pipeline, Data, BaseConfig

def main():
    """
    메인 함수로, 전체 파이프라인을 구성하고 실행합니다.
    """
    # 설정 정보를 가져옵니다.
    config = BaseConfig.get_config()

    # 탐지기(detectors) 생성
    detectors = [
        FaceDetectorFactory.create('dlib', config['dlib']),
        FaceDetectorFactory.create('yolo', config['yolo']),
        FaceDetectorFactory.create('mtcnn')
    ]
    
    predictors = [
        FacePredictorFactory.create('fairface', config['fairface'])
    ]

    # 타겟 얼굴 인코딩 로드
    with open(config['pickle_path'], 'rb') as f:
        target_encodings = np.array(pickle.load(f))

    # 파이프라인 설정
    pipeline = Pipeline(
        FaceDetector(detectors),                    # 얼굴을 탐지하는 단계
        FaceEncoder(),                              # 얼굴 인코딩을 생성하는 단계
        TargetFaceMatcher(target_encodings),        # 타겟 얼굴과 매칭하는 단계
        FacePredictor(predictors),                  # 얼굴 속성(성별, 나이, 인종 등)을 예측하는 단계
        FaceInfoCounter(),                          # 예측 결과를 기반으로 통계를 수집하는 단계
        InfoDrawer(),                               # 얼굴에 텍스트와 사각형을 그리는 단계
        InfoWriter(),                               # 분석 정보를 이미지에 추가하는 단계
        ImageResizer(target_size=1000),             # 이미지를 리사이징하는 단계
        Saver(),                                    # 결과 이미지를 저장하는 단계
    )

    # 이미지 폴더에서 처리할 이미지 목록을 가져옵니다.
    image_list = [
        f for f in os.listdir(config['image_folder'])
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]

    # 각 이미지를 파이프라인을 통해 처리합니다.
    for image_name in image_list:
        image_path = os.path.join(config['image_folder'], image_name)
        logging.info(f"이미지 처리 시작: {image_path}")
        # 데이터 객체를 생성하여 이미지 경로를 설정합니다.
        data = Data(config, image_path)
        # 파이프라인을 실행하여 이미지를 처리합니다.
        pipeline.run(data)
        logging.info(f"이미지 처리 완료: {image_path}")

if __name__ == "__main__":
    main()
