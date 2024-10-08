# steps/fairface_predict_step.py

from abc import ABC, abstractmethod
import logging

from ..core.config import PipelineStep


class FacePredictor(ABC):
    """
    얼굴 인식 모델의 추상 기본 클래스입니다.
    """
    @abstractmethod
    def predict(self, face_image):
        """
        얼굴 이미지를 입력으로 받아 예측 결과를 반환합니다.

        Args:
            face_image (numpy array): 얼굴 이미지.

        Returns:
            dict 또는 None: 예측 결과 딕셔너리 또는 오류 시 None.
        """
        pass


class FairFace(FacePredictor):
    """
    FairFace 모델을 사용하여 얼굴 이미지에서 인종, 성별, 나이를 예측하는 클래스입니다.
    """
    # 클래스 레이블 정의
    RACE_LABELS = ['백인', '흑인', '아시아', '중동']
    GENDER_LABELS = ['남성', '여성']
    AGE_LABELS = ['영아', '유아', '10대', '20대', '30대', '40대', '50대', '60대', '70+']

    def __init__(self, model_path):
        """
        FairFace 모델을 초기화합니다.

        Args:
            model_path (str): 모델 파일의 경로.
        """
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        try:
            # 디바이스 설정 (GPU 사용 가능 여부 확인)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"FacePredict : FairFace 모델 로드 중: {model_path}")

            # ResNet34 모델 불러오기 및 출력 레이어 수정
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 18)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = model.to(self.device).eval()

            # 이미지 전처리 파이프라인 정의
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )
            ])

            logging.info("FacePredict : FairFace 모델 로드 완료")
        except Exception as e:
            logging.error(f"FacePredict : FairFace 모델 로드 중 오류 발생: {e}")
            self.model = None

    def predict(self, face_image):
        """
        얼굴 이미지에서 인종, 성별, 나이를 예측합니다.

        Args:
            face_image (numpy array): 얼굴 이미지.

        Returns:
            dict 또는 None: 예측 결과 딕셔너리 또는 오류 시 None.
        """
        if self.model is None:
            logging.error("FairFace 모델이 로드되지 않았습니다.")
            return None

        import numpy as np
        import torch

        try:
            # 이미지 전처리 및 배치 차원 추가
            face_image = self.transform(face_image).unsqueeze(0).to(self.device)
        except Exception as e:
            logging.error(f"FacePredict : 이미지 전처리 중 오류 발생: {e}")
            return None

        with torch.no_grad():
            # 모델 추론 수행
            outputs = self.model(face_image).cpu().numpy().squeeze()

        # 예측 결과 추출
        race_pred_idx = np.argmax(outputs[:4])
        gender_pred_idx = np.argmax(outputs[7:9])
        age_pred_idx = np.argmax(outputs[9:18])

        race_pred = self.RACE_LABELS[race_pred_idx]
        gender_pred = self.GENDER_LABELS[gender_pred_idx]
        age_pred = self.AGE_LABELS[age_pred_idx]

        logging.info(f"FacePredict : FairFace 예측결과((({race_pred}, {gender_pred}, {age_pred})))")
        return {
            'race': race_pred,
            'gender': gender_pred,
            'age': age_pred
        }


class FacePredictorFactory:
    """
    FacePredictor 인스턴스를 생성하는 팩토리 클래스입니다.
    """
    @staticmethod
    def create(predictor_type, model_path=None):
        """
        지정된 유형의 얼굴 예측 모델을 생성합니다.

        Args:
            predictor_type (str): 예측기 유형 ('fairface' 등).
            model_path (str, optional): 모델 파일의 경로.

        Returns:
            FacePredictor: 생성된 얼굴 예측 모델 인스턴스.

        Raises:
            ValueError: 지원하지 않는 예측기 유형인 경우.
        """
        if predictor_type == 'fairface':
            return FairFace(model_path)
        else:
            raise ValueError(f"지원하지 않는 예측기 유형: {predictor_type}")


class FacePredictor(PipelineStep):
    """
    얼굴 이미지에서 인종, 성별, 나이를 예측하는 파이프라인 단계입니다.
    """
    def __init__(self, predictors):
        """
        FacePredict를 초기화합니다.

        Args:
            predictors (list): 사용할 FacePredictor 인스턴스들의 리스트.
        """
        self.predictors = predictors

    def process(self, data):
        """
        데이터를 처리하여 얼굴 속성을 예측합니다.
        
        Vars:
            image_rgb = data.image_rgb : 얼굴 이미지
            faces = data.predictions : 예측 정보 Dict
            faces = data.predictions.get('face_box', []) : 얼굴 박스 정보

        Args:
            data: 파이프라인 데이터 객체. 이미지와 얼굴 박스 정보를 포함해야 합니다.

        Returns:
            data: 예측 결과가 추가된 데이터 객체.
        """
        image_rgb = data.image_rgb
        faces = data.predictions.get('face_boxes', [])

        # 각 속성별로 예측 결과를 저장할 리스트 초기화
        race_predictions = []
        gender_predictions = []
        age_predictions = []

        if faces:
            for face in faces:
                x, y, x2, y2 = face
                # 얼굴 이미지 추출
                face_image = image_rgb[y:y2, x:x2]

                # 각 얼굴에 대한 예측 결과 저장
                face_result = {}

                for predictor in self.predictors:
                    result = predictor.predict(face_image)
                    if result is not None:
                        face_result.update(result)
                    else:
                        logging.warning("FacePredict : 예측 결과가 없습니다.")

                # 예측 결과를 각 리스트에 추가
                race_predictions.append(face_result.get('race', None))
                gender_predictions.append(face_result.get('gender', None))
                age_predictions.append(face_result.get('age', None))
            """
            Type:
            data.predictions = Dict{
                Str'': List[Str'', Str'', ...],
                Str'': List[Str'', Str'', ...],
                Str'': List[Str'', Str'', ...]
            }
            data.predictions = {
                'race': ['백인', '아시아', ...],
                'gender': ['남성', '여성', ...],
                'age': ['30대', '20대', ...]
            }
            
            현제:
            data.predictions = {
                'face_box': [(x1, y1, x2, y2), (x1, y1, x2, y2), ...],
                'race': ['백인', '아시아', ...],
                'gender': ['남성', '여성', ...],
                'age': ['30대', '20대', ...]
            }
            """
            # 예측 결과를 데이터 객체에 추가
            data.predictions.update({
                'race': race_predictions,
                'gender': gender_predictions,
                'age': age_predictions
            })

            logging.info(f"FacePredict : 총 {len(faces)}개의 얼굴 속성 예측 완료")
        else:
            logging.info("FacePredict : 검출된 얼굴이 없습니다.")

        return data
