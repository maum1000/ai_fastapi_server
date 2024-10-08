# config.py
import os
from pathlib import Path
import warnings
import logging
from abc import ABC, abstractmethod
import cv2


def setup_logging():
    """
    로깅 설정을 초기화하는 함수입니다.
    경고 메시지를 무시하고, 로깅 레벨과 형식을 설정합니다.
    """
    # 모든 경고를 무시합니다.
    warnings.filterwarnings("ignore")
    # 로깅 설정을 초기화합니다.
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s', 
        handlers=[logging.StreamHandler()] 
    )

class BaseConfig:
    setup_logging()  # 로깅 설정을 초기화합니다.
    # 클래스 변수로 기본 설정을 정의합니다.
    base_dir = os.path.join(Path(__file__).resolve().parent.parent, 'ai_files')
    dlib = os.path.join(base_dir, 'ai_models', 'DlibCNN', 'mmod_human_face_detector.dat')
    yolo_path = 'yolov8_n_face.pt'
    yolo = os.path.join(base_dir, 'ai_models','YOLOv8')
    fairface = os.path.join(base_dir, 'ai_models', 'FairFace', 'resnet34_fair_face_4.pt')
    image_folder = os.path.join(base_dir, 'image_test', 'test_park_mind_problem')
    pickle_path = os.path.join(base_dir, 'embeddings', 'FaceRecognition(ResNet34).pkl')
    font_path = os.path.join(base_dir, 'fonts', 'NanumGothic.ttf')
    results_folder =  os.path.join(Path(base_dir).resolve(), 'result_test') # 기본 값

    @classmethod
    def get_config(cls):
        return {
            "dlib": cls.dlib,
            "yolo": os.path.join(cls.yolo, cls.yolo_path),
            "fairface": cls.fairface,
            "image_folder": cls.image_folder,
            "pickle_path": cls.pickle_path,
            "font_path": cls.font_path,
            "results_folder": cls.results_folder  # 클래스 변수를 사용
        }

class PipelineStep(ABC):
    @abstractmethod
    def process(self, data):
        pass


class Pipeline:
    def __init__(self, *steps):
        self.steps = list(steps)

    def add(self, step):
        self.steps.append(step)
        return self  # 메서드 체이닝을 위해 self 반환

    def run(self, data):
        for step in self.steps:
            data = step.process(data)
        return data


class Data:
    def __init__(self, config, image_path):
        self.config = config
        self.image_path = image_path
        
        # OpenCV를 사용하여 이지 로드
        image = cv2.imread(self.image_path)
        if image is None:
            # 이미지 로드 실패 시 예외 발생
            raise ValueError(f"이미지를 로드할 수 없습니다: {self.image_path}")
        
        # BGR 이미지를 RGB로 변환
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.encodings = None 
        self.predictions = {}
        self.scale = None
        self.top = None
        self.left = None
        self.is_target_list = []
        self.output_image_path = None
        self.image_rgb

