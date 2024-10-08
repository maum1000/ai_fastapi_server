import os
import pickle
import numpy as np
from pathlib import Path
from ..ai_system.for_django.face_detection_system import (
    FaceDetectionManager,
    DlibFaceDetector,
    YOLOFaceDetector,
    MTCNNFaceDetector,
    FairFacePredictor,
    ImageProcessor,
    ImageMetadataManager,
    ForDjango,
    setup_warnings_and_logging,
)
#
def django_image_process(image_path):
    setup_warnings_and_logging()
    base_dir = Path(__file__).resolve().parent.parent.parent
    config = {
        "dlib_model_path"       : os.path.join(base_dir, 'ai_system',   'Model',     'DilbCNN',  'mmod_human_face_detector.dat'),
        "yolo_model_path"       : os.path.join(base_dir, 'ai_system',   'Model',     'YOLOv8',   'yolov8n-face.pt'),
        "fair_face_model_path"  : os.path.join(base_dir, 'ai_system',   'Model',     'FairFace', 'resnet34_fair_face_4.pt'),
        "image_folder"          : os.path.join(base_dir, 'ai_system',   'Image',     'test',     'test_park_mind_problem'),
        "results_folder"        : os.path.join(base_dir, 'pybo'     ,   'media',     'answer_image'),
        "pickle_path"           : os.path.join(base_dir, 'ai_system',   'Embedings', 'FaceRecognition(ResNet34).pkl'),
        "font_path"             : os.path.join(base_dir, 'ai_system',   'fonts',     'NanumGothic.ttf'),
    }
    #
    # 얼굴 탐지기, 예측기, 이미지 프로세서, 메타데이터 관리자 생성
    detector_manager = FaceDetectionManager([
        DlibFaceDetector(config['dlib_model_path']),
        YOLOFaceDetector(config['yolo_model_path']),
        MTCNNFaceDetector()
    ])
    # 얼굴 예측기 생성
    predictor = FairFacePredictor(config['fair_face_model_path'])
    # 이미지 프로세서, 메타데이터 관리자 생성
    image_processor = ImageProcessor()
    metadata_manager = ImageMetadataManager()
    # 얼굴 인식 시스템 객체 생성
    face_recognition_system = ForDjango(config, detector_manager, predictor, image_processor, metadata_manager)
    #
    # 타겟 얼굴 인코딩 로드
    with open(config['pickle_path'], 'rb') as f:
        target_encodings = pickle.load(f)
        #
    #
    # 얼굴 인식 및 처리
    output_path = face_recognition_system.process_image(image_path, target_encodings)
    return output_path # 이미지 경로 반환

