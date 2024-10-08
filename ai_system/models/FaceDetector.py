# steps/detect_faces_step.py

import numpy as np
import logging
from abc import ABC, abstractmethod

from ..core.config import PipelineStep

class FaceDetectModel(ABC):
    """
    얼굴 검출 모델의 추상 클래스입니다.
    """
    @abstractmethod
    def detect_faces(self, image):
        """
        이미지에서 얼굴을 검출하는 메서드입니다.

        Args:
            image: 얼굴을 검출할 이미지.

        Returns:
            faces: 검출된 얼굴 좌표 리스트.
        """
        pass

class Dlib(FaceDetectModel):
    """
    Dlib 라이브러리를 사용하는 얼굴 검출 모델입니다.
    """
    def __init__(self, model_path):
        """
        Dlib 모델을 초기화합니다.

        Args:
            model_path: Dlib 모델 파일의 경로.
        """
        import dlib
        try:
            logging.info(f"FaceDetector : Dlib 모델 로드 중: {model_path}")
            self.detector = dlib.cnn_face_detection_model_v1(model_path)
        except Exception as e:
            logging.error(f"FaceDetector : Dlib 모델 로드 중 오류 발생: {e}")
            self.detector = None

    def detect_faces(self, image):
        """
        이미지에서 얼굴을 검출합니다.

        Args:
            image: 얼굴을 검출할 이미지.

        Returns:
            faces: 검출된 얼굴 좌표 리스트.
        """
        if self.detector is None:
            logging.error("FaceDetector : Dlib 모델이 로드되지 않았습니다.")
            return []
        detections = self.detector(image, 1)
        faces = [
            (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
            for d in detections
        ]
        return faces

class YOLO(FaceDetectModel):
    """
    YOLO 모델을 사용하는 얼굴 검출 모델입니다.
    """
    def __init__(self, model_path):
        """
        YOLO 모델을 초기화합니다.

        Args:
            model_path: YOLO 모델 파일의 경로.
        """
        from ultralytics import YOLO
        try:
            logging.info(f"FaceDetector : YOLO 모델 로드 중: {model_path}")
            self.detector = YOLO(model_path)
        except Exception as e:
            logging.error(f"FaceDetector : YOLO 모델 로드 중 오류 발생: {e}")
            self.detector = None

    def detect_faces(self, image_path):
        """
        이미지에서 얼굴을 검출합니다.

        Args:
            image_path: 얼굴을 검출할 이미지의 경로.

        Returns:
            faces: 검출된 얼굴 좌표 리스트.
        """
        if self.detector is None:
            logging.error("FaceDetector : YOLO 모델이 로드되지 않았습니다.")
            return []
        results = self.detector.predict(image_path, conf=0.35, imgsz=640, max_det=1000)
        faces = [
            (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3])
            )
            for result in results for box in result.boxes
        ]
        return faces

class MTCNN(FaceDetectModel):
    """
    MTCNN 라이브러리를 사용하는 얼굴 검출 모델입니다.
    """
    def __init__(self):
        """
        MTCNN 모델을 초기화합니다.
        """
        from mtcnn import MTCNN
        try:
            logging.info("FaceDetector : MTCNN 모델 로드 중...")
            self.detector = MTCNN()
        except Exception as e:
            logging.error(f"FaceDetector : MTCNN 모델 로드 중 오류 발생: {e}")
            self.detector = None

    def detect_faces(self, image):
        """
        이미지에서 얼굴을 검출합니다.

        Args:
            image: 얼굴을 검출할 이미지.

        Returns:
            faces: 검출된 얼굴 좌표 리스트.
        """
        if self.detector is None:
            logging.error("FaceDetector : MTCNN 모델이 로드되지 않았습니다.")
            return []
        detections = self.detector.detect_faces(image)
        faces = [
            (
                f['box'][0],
                f['box'][1],
                f['box'][0] + f['box'][2],
                f['box'][1] + f['box'][3]
            )
            for f in detections
        ]
        return faces

class FaceDetectorFactory:
    """
    얼굴 검출 모델의 인스턴스를 생성하는 팩토리 클래스입니다.
    """
    @staticmethod
    def create(detector_type, model_path=None):
        """
        지정된 유형의 얼굴 검출 모델을 생성합니다.

        Args:
            detector_type: 사용할 검출기 유형 ('dlib', 'yolo', 'mtcnn').
            model_path: 모델 파일의 경로 (필요한 경우).

        Returns:
            FaceDetectModel의 인스턴스.

        Raises:
            ValueError: 지원하지 않는 탐지기 유형인 경우.
        """
        if detector_type == 'dlib':
            return Dlib(model_path)
        elif detector_type == 'yolo':
            return YOLO(model_path)
        elif detector_type == 'mtcnn':
            return MTCNN()
        else:
            raise ValueError(f"지원하지 않는 탐지기 유형: {detector_type}")

class FaceDetector(PipelineStep):
    """
    여러 얼굴 검출 모델을 사용하여 얼굴을 검출하는 파이프라인 단계입니다.
    """
    def __init__(self, detectors):
        """
        FaceDetector를 초기화합니다.

        Args:
            detectors: 사용할 얼굴 검출 모델의 리스트.
        """
        self.detectors = detectors

    def process(self, data):
        """
        이미지를 처리하여 얼굴을 검출합니다.

        Args:
            data: 파이프라인 데이터 객체.

        Vars:
            image_rgb = data.image_rgb : 얼굴 이미지
            image_path = data.image_path : 이미지 경로

        Returns:
            처리된 데이터 객체.
        """
        image_rgb = data.image_rgb
        image_path = data.image_path
        all_faces = []

        for detector in self.detectors:
            try:
                # YOLO 모델의 경우 이미지 경로를 사용
                if isinstance(detector, YOLO):
                    faces = detector.detect_faces(image_path)
                else:
                    faces = detector.detect_faces(image_rgb)
                logging.info(f"FaceDetector : {detector.__class__.__name__}: {len(faces)}개의 얼굴 검출")
                all_faces.extend(faces)
            except Exception as e:
                logging.error(f"FaceDetector : {detector.__class__.__name__} 탐지 중 오류 발생: {e}")
        """
        all_faces = List[Tuple(), Tuple(), ...]
                    [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
                    [(left, top, right, bottom), (left, top, right, bottom), ...]
                    [(얼굴1), (얼굴2), ...]
                    
        nms_faces = List[Tuple(), Tuple(), ...]
                    마찬가지...
        """
        # 중복된 얼굴 박스를 제거하기 위해 NMS 적용
        nms_faces = self._apply_non_max_suppression(all_faces)
        """
        data.predictions = {
            'face_boxes': all_faces
        }
        """
        data.predictions['face_boxes'] = nms_faces # List
        logging.info(f"FaceDetector : NMS 후 탐지된 얼굴 수: {len(nms_faces)}")
        return data

    @staticmethod
    def _apply_non_max_suppression(faces, overlap_thresh=0.3):
        """
        Non-Max Suppression을 적용하여 중복된 얼굴 박스를 제거합니다.

        Args:
            faces: 얼굴 박스의 리스트.
            overlap_thresh: 박스가 겹치는 허용 임계값.

        Returns:
            중복이 제거된 얼굴 박스의 리스트.
        """
        if len(faces) == 0:
            return []

        boxes = np.array(faces)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = y2.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 다른 박스들과의 교집합 계산
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # IoU 계산
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 임계값 이하인 박스만 남김
            inds = np.where(ovr <= overlap_thresh)[0]
            order = order[inds + 1]

        return boxes[keep].astype(int).tolist()
