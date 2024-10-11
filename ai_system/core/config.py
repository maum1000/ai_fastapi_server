import os
from pathlib import Path
import warnings
import logging
from abc import ABC, abstractmethod
import cv2
from .utils import ImageUtils


def setup_logging():
    """
    로깅 설정을 초기화하는 함수입니다.
    
    경고 메시지를 무시하고, 로깅 레벨과 형식을 설정합니다.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # 모든 경고를 무시합니다.
    warnings.filterwarnings("ignore")
    
    # 로깅 설정을 초기화합니다.
    logging.basicConfig(
        level=logging.INFO,  # 로깅 레벨을 INFO로 설정
        format='%(asctime)s [%(levelname)s] %(message)s',  # 로그 형식 정의
        handlers=[logging.StreamHandler()]  # 로그 출력 방식 (콘솔 출력)
    )


class BaseConfig:
    """
    AI 모델 및 파일 경로 설정을 관리하는 클래스입니다.

    이 클래스는 AI 모델, 이미지, 결과 파일 등의 경로를 클래스 변수로 정의하고,
    설정 값을 딕셔너리로 반환하는 기능을 제공합니다.

    Attributes
    ----------
    base_dir : str
        AI 파일들이 저장된 기본 디렉터리 경로입니다.
    dlib : str
        Dlib 모델 경로입니다.
    yolo : str
        YOLO 모델 경로입니다.
    yolo_path : str
        YOLO 모델 파일 이름입니다.
    fairface : str
        FairFace 모델 경로입니다.
    image_folder : str
        테스트에 사용할 이미지가 저장된 폴더 경로입니다.
    pickle_path : str
        얼굴 인식 ResNet34 모델이 저장된 경로입니다.
    font_path : str
        폰트 파일 경로입니다.
    results_folder : str
        테스트 결과가 저장될 폴더 경로입니다.

    Methods
    -------
    get_config():
        설정 정보를 딕셔너리 형태로 반환합니다.
    """
    
    setup_logging()  # 로깅 설정을 초기화합니다.

    # 클래스 변수로 경로 설정
    base_dir = os.path.join(Path(__file__).resolve().parent.parent, 'ai_files')
    dlib = os.path.join(base_dir, 'ai_models', 'DlibCNN', 'mmod_human_face_detector.dat')
    yolo_president_path = 'yolov8_l_trump.pt'
    yolo_face_path = 'yolov8_n_face.pt'
    yolo = os.path.join(base_dir, 'ai_models','YOLOv8')
    fairface = os.path.join(base_dir, 'ai_models', 'FairFace', 'resnet34_fair_face_4.pt')
    image_folder = os.path.join(base_dir, 'image_test', 'test_park_mind_problem')
    pickle_path = os.path.join(base_dir, 'embeddings', 'FaceRecognition(ResNet34).pkl')
    font_path = os.path.join(base_dir, 'fonts', 'NanumGothic.ttf')
    results_folder = os.path.join(Path(base_dir).resolve(), 'result_test')

    @classmethod
    def get_config(cls) -> dict:
        """
        설정 정보를 딕셔너리 형태로 반환하는 클래스 메서드입니다.

        Returns
        -------
        dict
            설정 정보를 담고 있는 딕셔너리입니다.
        """
        return {
            "dlib": cls.dlib,
            "yolo": os.path.join(cls.yolo, cls.yolo_face_path),
            "yolo_president": os.path.join(cls.yolo, cls.yolo_president_path),
            "fairface": cls.fairface,
            "image_folder": cls.image_folder,
            "pickle_path": cls.pickle_path,
            "font_path": cls.font_path,
            "results_folder": cls.results_folder
        }


class PipelineStep(ABC):
    """
    파이프라인 단계의 추상 클래스로, 각 단계에서 실행할 메서드를 정의합니다.

    Methods
    -------
    process(data):
        각 파이프라인 단계에서 실행할 메서드. 구체적인 단계에서는 이 메서드를 구현해야 합니다.
    """
    
    @abstractmethod
    def process(self, data):
        """
        각 파이프라인 단계에서 데이터를 처리하는 메서드입니다.

        Parameters
        ----------
        data : any
            처리할 데이터입니다.

        Returns
        -------
        any
            처리된 데이터를 반환합니다.
        """
        pass


class Pipeline:
    """
    여러 단계를 처리하는 파이프라인을 관리하는 클래스입니다.

    Attributes
    ----------
    steps : list
        파이프라인에 추가된 단계 리스트입니다.

    Methods
    -------
    add(step):
        파이프라인에 단계를 추가합니다.
    run(data):
        파이프라인을 실행하여 데이터를 처리합니다.
    """
    
    def __init__(self, *steps):
        """
        파이프라인을 초기화하는 생성자입니다.

        Parameters
        ----------
        steps : tuple
            초기 파이프라인 단계들입니다.
        """
        self.steps = list(steps)

    def add(self, step) -> 'Pipeline':
        """
        파이프라인에 단계를 추가합니다.

        Parameters
        ----------
        step : PipelineStep
            추가할 파이프라인 단계입니다.

        Returns
        -------
        Pipeline
            파이프라인 객체를 반환하여 메서드 체이닝이 가능하도록 합니다.
        """
        self.steps.append(step)
        return self

    def run(self, data):
        """
        파이프라인을 실행하여 데이터를 순차적으로 처리합니다.

        Parameters
        ----------
        data : any
            파이프라인에 입력할 데이터입니다.

        Returns
        -------
        any
            최종 처리된 데이터를 반환합니다.
        """
        for step in self.steps:
            data = step.process(data)
        return data


class Data:
    """
    파이프라인에서 사용할 데이터 객체입니다.
    이미지 처리와 관련된 데이터와 설정 정보를 관리합니다.

    Attributes
    ----------
    config : dict
        설정 정보 딕셔너리입니다.
    image_path : str
        이미지 파일 경로입니다.
    image_rgb : numpy.ndarray
        RGB 형식으로 변환된 이미지 데이터입니다.
    encodings : list or None
        이미지에서 추출된 인코딩 정보입니다. (초기값: None)
    predictions : dict
        모델 예측 결과를 저장하는 딕셔너리입니다.
    scale : float or None
        이미지 스케일 값입니다. (초기값: None)
    top : int or None
        이미지에서 탐지된 객체의 상단 위치입니다. (초기값: None)
    left : int or None
        이미지에서 탐지된 객체의 좌측 위치입니다. (초기값: None)
    is_target_list : list
        탐지된 객체가 목표인지 여부를 저장하는 리스트입니다. (초기값: 빈 리스트)
    output_image_path : str or None
        처리된 결과 이미지의 저장 경로입니다. (초기값: None)
    """
    
    def __init__(self, config: dict, image_path: str):
        """
        Data 객체를 초기화하는 생성자입니다.

        Parameters
        ----------
        config : dict
            설정 정보 딕셔너리입니다.
        image_path : str
            이미지 파일 경로입니다.
        """
        self.config = config
        self.image_path = image_path
        
        # OpenCV를 사용하여 이미지 로드
        image = cv2.imread(self.image_path)
        if image is None:
            # 이미지 로드 실패 시 예외 발생
            raise ValueError(f"이미지를 로드할 수 없습니다: {self.image_path}")
        
        # BGR 이미지를 RGB로 변환
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 기본값 설정
        self.encodings = None 
        self.predictions = {}
        self.scale = None
        self.top = None
        self.left = None
        self.is_target_list = []
        self.output_image_path = None
        self.image_utils = ImageUtils(self.config)
        self.president_name_list = []
