import logging
import os
import cv2
import dlib
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import face_recognition
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO
from mtcnn import MTCNN
import time
import piexif
from abc import ABC, abstractmethod
import io
import shutil
from pathlib import Path
import warnings
import json
# config.json 경로를 Django settings에서 가져오기
from django.conf import settings
#
# =========================
# 이미지 처리 함수들
# =========================
#
def resize_image_with_padding(image, target_size):
    """이미지를 리사이즈하고 패딩을 추가하는 함수"""
    #
    h, w, _ = image.shape
    scale = target_size / max(h, w)
    resized_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    #
    delta_w = target_size - resized_img.shape[1]
    delta_h = target_size - resized_img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    #
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #
    return new_img, scale, top, left
    #
#
def draw_korean_text(config, image, text, position, font_size, font_color=(255, 255, 255), background_color=(0, 0, 0)):
    """이미지에 한글 텍스트를 그리는 함수"""
    #
    font_path = config['font_path']
    #
    # 텍스트가 없으면 이미지 그대로 반환
    if text == '':
        return image
        #
    #
    font = ImageFont.truetype(font_path, int(font_size))
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    #
    # 텍스트 크기 측정
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    #
    # 텍스트에 맞춰 박스 크기 조정
    box_x0 = position[0] - 5
    box_y0 = position[1] - 5
    box_x1 = position[0] + text_width + 5
    box_y1 = position[1] + text_height + 5
    #
    # 이미지가 텍스트를 수용할 수 있도록 크기를 확장
    if box_x1 > image.shape[1] or box_y1 > image.shape[0]:
        new_width = max(box_x1, image.shape[1])
        new_height = max(box_y1, image.shape[0])
        extended_img = np.ones((new_height, new_width, 3), dtype=np.uint8) * 0  # 흰색 배경
        extended_img[:image.shape[0], :image.shape[1]] = image
        img_pil = Image.fromarray(extended_img)
        draw = ImageDraw.Draw(img_pil)
        #
    #
    # 배경 박스 그리기
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=background_color)
    #
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=font_color)
    #
    return np.array(img_pil)
    #
#
def extend_image_with_text(config, image, text, font_size, font_color=(255, 255, 255), background_color=(0, 0, 0)):
    """이미지 확장 및 텍스트 추가 함수 (위쪽 확장)"""
    #
    font_path = config['font_path']
    # 이미지 크기 가져오기
    height, width, _ = image.shape
    #
    # 텍스트 크기 계산
    font = ImageFont.truetype(font_path, font_size)
    line_spacing = int(font_size * 1.5)
    text_lines = text.count('\n') + 1
    total_text_height = line_spacing * text_lines
    #
    # 새 이미지 생성 (텍스트를 위한 공간 + 원본 이미지)
    extended_image = np.zeros((height + total_text_height + 20, width, 3), dtype=np.uint8)  # 검은색 배경
    extended_image[total_text_height + 20:, 0:width] = image
    #
    # 텍스트 추가
    extended_image_pil = Image.fromarray(extended_image)
    draw = ImageDraw.Draw(extended_image_pil)
    draw.rectangle([(0, 0), (width, total_text_height + 20)], fill=background_color)
    draw.text((10, 10), text, font=font, fill=font_color)
    #
    return np.array(extended_image_pil)
    #
#
def copy_image_and_add_metadata(image_path, output_folder):
    """ 이미지 복사 및 메타데이터 추가 함수 """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    # output_folder 경로에 이미지 복사
    shutil.copy(image_path, output_folder)
    # 복사된 이미지 경로
    copied_image_path = os.path.join(output_folder, os.path.basename(image_path))
    #
    # 복사된 이미지 연 후 메타데이터 추가
    with Image.open(copied_image_path) as meta_im:
        if meta_im.mode == 'RGBA':
            meta_im = meta_im.convert('RGB')
            #
        #
        thumb_im = meta_im.copy()
        o = io.BytesIO()
        thumb_im.thumbnail((50, 50), Image.Resampling.LANCZOS)
        thumb_im.save(o, "jpeg")
        thumbnail = o.getvalue()
        #
        zeroth_ifd = {
            piexif.ImageIFD.Make: u"oldcamera",
            piexif.ImageIFD.XResolution: (96, 1),
            piexif.ImageIFD.YResolution: (96, 1),
            piexif.ImageIFD.Software: u"piexif",
            piexif.ImageIFD.Artist: u"0!code",
        }
        #
        exif_ifd = {
            piexif.ExifIFD.DateTimeOriginal: u"2099:09:29 10:10:10",
            piexif.ExifIFD.LensMake: u"LensMake",
            piexif.ExifIFD.Sharpness: 65535,
            piexif.ExifIFD.LensSpecification: ((1, 1), (1, 1), (1, 1), (1, 1)),
        }
        #
        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSAltitudeRef: 1,
            piexif.GPSIFD.GPSDateStamp: u"1999:99:99 99:99:99",
        }
        #
        first_ifd = {
            piexif.ImageIFD.Make: u"oldcamera",
            piexif.ImageIFD.XResolution: (40, 1),
            piexif.ImageIFD.YResolution: (40, 1),
            piexif.ImageIFD.Software: u"piexif"
        }
        #
        exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": first_ifd, "thumbnail": thumbnail}
        exif_bytes = piexif.dump(exif_dict)
        #
        meta_im.save(copied_image_path, exif=exif_bytes)
        logging.info(f"이미지가 저장되었습니다. {copied_image_path}")
        #
    #
#
def print_image_exif_data(image_path):
    """ 이미지의 Exif 데이터 출력 함수 """
    with Image.open(image_path) as im:
        exif_data = piexif.load(im.info['exif'])
        print(exif_data)
        #
    #
#
# =========================
# 추상화: AI Model 인터페이스
# =========================
class AIModel(ABC):
    @abstractmethod
    def __init__(self, model_path):
        """
        모델을 로드하는 메서드, 
        
        try - except 구문을 사용하여 모델 로드 중 발생하는 예외를 로깅과 함께 처리하고,
        
        로드에 실패한 경우 None을 반환하도록 구현
        
        구체적 구현은 하위 클래스에서 제공
        
        try - except 구문이 중복되는 것을 감수하고, 추상화를 하였음
        
        """
        pass
        #
    #
    @ abstractmethod
    def predict(self, image, image_path=None):
        """
        
        각 모델들의 예측을 시작하는 메서드, 
        
        각 모델마다 반환하는 얼굴 좌표의 형식이 다르므로,
        
        추상화를 통해 통일된 형식으로 반환하도록 구현
        
        구체적 구현은 하위 클래스에서 제공
        
        """
        pass
        #
    #
#
# =========================
# Dlib 모델 Face Detector 구현
# =========================
class DlibFaceDetector(AIModel):
    def __init__(self, model_path):
        """Dlib 얼굴 탐지 모델 로드"""
        try:
            logging.info(f"Dlib 모델 로드 중: {model_path}")
            self.detector = dlib.cnn_face_detection_model_v1(model_path)
        except FileNotFoundError:
            logging.error(f"Dlib 모델 파일을 찾을 수 없습니다: {model_path}")
            self.detector = None
        except Exception as e:
            logging.error(f"Dlib 모델 로드 중 오류 발생: {e}")
            self.detector = None
            #
        #
    #
    def predict(self, image):
        """Dlib을 이용해 이미지에서 얼굴을 탐지"""
        if self.detector is None:
            logging.error("Dlib 모델이 로드되지 않았습니다.")
            return []
            #
        #
        return [(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()) for d in self.detector(image, 1)]
        #
    #
#
# =========================
# YOLO 모델 Face Detector 구현
# =========================
class YOLOFaceDetector(AIModel):
    def __init__(self, model_path):
        """YOLO 얼굴 탐지 모델 로드"""
        try:
            logging.info(f"YOLO 모델 로드 중: {model_path}")
            self.detector = YOLO(model_path)
        except FileNotFoundError:
            logging.error(f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
            self.detector = None
        except Exception as e:
            logging.error(f"YOLO 모델 로드 중 오류 발생: {e}")
            self.detector = None
            #
        #
    #
    def predict(self, image_path):
        """YOLO을 이용해 이미지에서 얼굴을 탐지"""
        if self.detector is None:
            logging.error("YOLO 모델이 로드되지 않았습니다.")
            return []
            #
        #
        results = self.detector.predict(image_path, conf=0.35, imgsz=1280, max_det=1000)
        return [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) for result in results for box in result.boxes]
        #
    #
#
# =========================
# MTCNN 모델 Face Detector 구현
# =========================
class MTCNNFaceDetector(AIModel):
    def __init__(self):
        """MTCNN 얼굴 탐지 모델 로드"""
        try:
            logging.info(f"MTCNN 모델 로드 중...")
            self.detector = MTCNN()
        except FileNotFoundError:
            logging.error(f"MTCNN 모델 파일을 찾을 수 없습니다!")
            self.detector = None
        except Exception as e:
            logging.error(f"MTCNN 모델 로드 중 오류 발생: {e}")
            self.detector = None
            #
        #
    #
    def predict(self, image):
        """MTCNN을 이용해 이미지에서 얼굴을 탐지"""
        if self.detector is None:
            logging.error("MTCNN 모델이 로드되지 않았습니다.")
            return []
            #
        #
        return [(f['box'][0], f['box'][1], f['box'][0] + f['box'][2], f['box'][1] + f['box'][3]) for f in self.detector.detect_faces(image)]
        #
    #
#
# =========================
# FairFace 모델 Face Predictor 구현 
# =========================
class FairFacePredictor(AIModel):
    def __init__(self, model_path):
        """FairFace 모델 로드"""
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"FairFace 모델 load 중:\n{model_path}")
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 18)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = model.to(self.device).eval()
            logging.info("FairFace 모델 load 완료")
        except Exception as e:
            logging.error(f"FairFace 모델 로드 중 오류 발생: {e}")
            self.model = None
            #
        #
    #
    def predict(self, face_image):
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        try:
            face_image = trans(face_image).unsqueeze(0).to(self.device)
        except ValueError:
            logging.error("이미지가 너무 작거나 손상됨, 예측 건너뜀.")
            return None
            #
        #
        with torch.no_grad():
            outputs = self.model(face_image).cpu().numpy().squeeze()
        #
        race_pred = np.argmax(outputs[:4])
        gender_pred = np.argmax(outputs[7:9])
        age_pred = np.argmax(outputs[9:18])
        #
        race_text = ['백인', '흑인', '아시아', '중동'][race_pred]
        gender_text, box_color = [('남성', (255, 100, 50)), ('여성', (50, 100, 255))][gender_pred]
        age_text = ['영아', '유아', '10대', '20대', '30대', '40대', '50대', '60대', '70+'][age_pred]
        #
        return {"race": race_text, "gender": gender_text, "box_color": box_color, "age": age_text}
        #
    #
#
# =========================
# 추상화: Model 관리자 클래스
# =========================
class ModelManager(ABC):
    @abstractmethod
    def __init__(self, model_path):
        """구체적 구현은 하위 클래스에서 제공"""
        pass
        #
    #
    @abstractmethod
    def manage_prediction(self, image, image_path=None):
        """구체적 구현은 하위 클래스에서 제공"""
        pass
        #
    #
#
# =========================
# FaceDetector 관리자 클래스
# =========================
class FaceDetectors(ModelManager):
    def __init__(self, *detectors):
        self.detectors = detectors

    def manage_prediction(self, image, image_path=None):
        """모든 탐지기를 사용해 얼굴을 탐지하고, 비최대 억제 적용"""
        logging.info("얼굴 탐지 시작...")
        all_faces = []
        for detector in self.detectors:
            try:
                if isinstance(detector, YOLOFaceDetector) and image_path: # YOLOFaceDetector의 경우 이미지 경로를 사용하여 탐지
                    faces =  detector.predict(image_path)
                else:
                    faces =  detector.predict(image)
            except Exception as e:
                logging.error(f"얼굴 탐지 중 오류 발생: {e}")
                raise
                #
            #
            all_faces.extend(faces)
            #
        #
        logging.info(f"총 {len(all_faces)}개의 얼굴 검출.")
        #
        # 비최대 억제 적용
        return self._apply_non_max_suppression(all_faces)
        #
    #
    @staticmethod
    def _apply_non_max_suppression(faces):
        """
        비최대 억제를 적용하여 중복 얼굴 영역을 제거
        """
        if len(faces) == 0:
            return []
            #
        #
        # 얼굴 영역 좌표 비최대 억제 로직
        boxes = np.array(faces).astype("float")
        pick = []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        #
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            #
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            #
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            #
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.3)[0])))
            #
        #
        return boxes[pick].astype("int").tolist()
        #
    #
#
# =========================
# FacePredictor 관리자 클래스
# =========================
class FacePredictors(ModelManager):
    def __init__(self, *predictors):
        self.predictors = predictors

    def manage_prediction(self, image):
        """FacePredictor의 예측 결과를 관리"""
        logging.info("얼굴 예측 시작...")
        all_predictions = {}
        for predictor in self.predictors:
            try:
                prediction = predictor.predict(image)
            except Exception as e:
                logging.error(f"예측 중 오류 발생: {e}")
                continue
            all_predictions.update(prediction)
        logging.info(f"예측 결과: {all_predictions}")
        return all_predictions
        #
    #
#
# =========================
# 얼굴 인식 시스템 클래스
# 기존의 advanced project 의 로직
# =========================
class FaceRecognitionSystem:
    def __init__(self, config, detector_manager, predictor_manager):
        self.config = config
        self.detector_manager = detector_manager
        self.predictor_manager = predictor_manager
        #
    #
    def process_image(self, image_path, target_encodings):
        """이미지에서 얼굴을 탐지하고 결과를 저장"""
        try:
            image_rgb, faces = self._detect_faces(image_path) # 얼굴 탐지
            predictions, face_cnt, race_cnt, male_cnt = self._fairface_predict(image_rgb, faces, target_encodings) # 얼굴 예측
            result_image = self._draw_results(image_rgb, predictions, face_cnt, male_cnt, race_cnt) # 결과 그리기
            self._save_results(image_path, result_image, predictions) # 결과 저장
        except Exception as e:
            logging.error(f"이미지 처리 중 오류 발생: {e}")
            #
        #
    #
    def _detect_faces(self, image_path):
        """이미지에서 얼굴을 탐지"""
        try:
            image = cv2.imread(image_path) # 이미지 읽기
            #
            # 이미지 읽기 실패 시 예외 발생
            if image is None: 
                raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
                #
            #
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB로 변환
            faces = self.detector_manager.manage_prediction(image_rgb, image_path) # 얼굴 탐지
            logging.info(f"얼굴 탐지 완료: {len(faces)}명")
            #
            return image_rgb, faces # RGB 이미지와 얼굴 좌표 반환
            #
        #
        except Exception as e:
            logging.error(f"얼굴 탐지 중 오류 발생: {e}")
            raise
            #
        #
    #
    def _fairface_predict(self, image_rgb, faces, target_encodings):
        """얼굴을 예측 결과를 잘 추합해서 반환"""
        predictions = []
        face_cnt = 0
        race_cnt = {'백인': 0, '흑인': 0, '아시아': 0, '중동': 0}
        male_cnt = 0
        #
        for face in faces:
            prediction = self._process_single_face(image_rgb, face, target_encodings)
            if prediction:
                predictions.append(prediction)
                face_cnt += 1
                race_text, gender_text = prediction[4], prediction[5]
                race_cnt[race_text] += 1
                if gender_text == '남성':
                    male_cnt += 1
                    #
                #
            #
        #
        return predictions, face_cnt, race_cnt, male_cnt
        #
    #
    def _process_single_face(self, image_rgb, face, target_encodings):
        """단일 얼굴에 대해 예측 수행"""
        try:
            x, y, x2, y2 = face # 얼굴 좌표
            face_width, face_height = x2 - x, y2 - y # 얼굴 크기
            face_image = image_rgb[y:y2, x:x2] # 얼굴 이미지
            encodings = face_recognition.face_encodings(image_rgb, [(y, x + face_width, y + face_height, x)]) # 얼굴 인코딩
            #
            # 얼굴 인코딩 실패 시 예외 발생
            if not encodings:
                logging.warning(f"얼굴 인코딩 실패: {face}")
                return None
                #
            #   
            prediction_result = self.predictor_manager.manage_prediction(face_image) # 얼굴 예측
            race_text = prediction_result.get("race", "알 수 없음") # 인종
            gender_text = prediction_result.get("gender", "알 수 없음") # 성별
            box_color = prediction_result.get("box_color", (0, 0, 0)) # 박스 색상
            age_text = prediction_result.get("age", "알 수 없음") # 나이
            #
            # 예측 결과 텍스트
            is_gaka = any(face_recognition.compare_faces(target_encodings, encodings[0], tolerance=0.3))
            prediction_text = '가카!' if is_gaka and gender_text == '남성' else age_text
            #
            return x, y, x2 - x, y2 - y, race_text, gender_text, box_color, prediction_text
            #
        #
        except Exception as e:
            logging.error(f"단일 얼굴 처리 중 오류 발생: {e}")
            return None
            #
        #
    #
    def _draw_results(self, image_rgb, predictions, face_cnt, male_cnt, race_cnt):
        """결과를 이미지에 그린 후 리턴"""
        font_size = max(12, int(image_rgb.shape[1] / 200)) # 폰트 크기
        image_rgb, scale, top, left = resize_image_with_padding(image_rgb, 512) # 이미지 리사이즈
        #
        # 예측 결과 그리기
        for x, y, w, h, _, _, box_color, prediction_text in predictions: 
            x = int(x * scale) + left
            y = int(y * scale) + top
            w = int(w * scale)
            h = int(h * scale)
            image_rgb = draw_korean_text(self.config, image_rgb, prediction_text, (x, y), 15, font_color=(0, 0, 0), background_color=box_color)
            image_rgb = cv2.rectangle(image_rgb, (x, y), (x + w, y + h), box_color, 2)
            #
        #
        info_text = f"검출된 인원 수: {face_cnt}명\n남성: {male_cnt}명\n여성: {face_cnt - male_cnt}명\n"
        race_info = "\n".join([f"{race}: {count}명" for race, count in race_cnt.items() if count > 0])
        image_rgb = extend_image_with_text(self.config, image_rgb, info_text + race_info, font_size)
        #
        return image_rgb
        #
    #
    def _save_results(self, image_path, image_rgb, predictions):
        """결과 이미지를 저장하고 메타데이터 추가"""
        try:
            output_path = os.path.join(self.config['results_folder'], os.path.basename(image_path)) # 결과 이미지 경로
            cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)) # 이미지 저장
            logging.info(f"이미지 분석 결과 저장: {output_path}") 
            #
            gaka_detected = any("가카" in pred[7] for pred in predictions) # 가카 여부
            detection_folder = "detection_target" if gaka_detected else "detection_non_target" # 타겟 여부에 따라 폴더 설정
            output_folder = os.path.join(self.config['results_folder'], detection_folder) # 결과 폴더
            #
            copy_image_and_add_metadata(image_path, output_folder) # 이미지 복사 및 메타데이터 추가
            #
            logging.info(f"메타데이터 추가된 이미지 저장: {output_folder}") 
        except Exception as e:
            logging.error(f"결과 저장 중 오류 발생: {e}")
        #
    #
#
# =========================
# 얼굴 인식 시스템 클래스 for Django
# =========================
class ForDjango(FaceRecognitionSystem):
    def __init__(self, config, detector_manager, predictor_manager):
        super().__init__(config, detector_manager, predictor_manager)
        #
    #
    def process_image(self, image_path, target_encodings):
        """
        이미지에서 얼굴을 탐지하고 결과를 저장
        image_path : Django에서 전달받은 이미지 경로(pybo/pybo/media/image/파일명)
        output_path : 결과 이미지 경로(pybo/pybo/media/answer_image/파일명)
        """
        image_rgb, faces = self._detect_faces(image_path)
        predictions, face_cnt, race_cnt, male_cnt = self._predict_faces(image_rgb, faces, target_encodings)
        result_image = self._draw_results(image_rgb, predictions, face_cnt, male_cnt, race_cnt)
        output_path = self._save_results(image_path, result_image, predictions) 
        django_path = os.path.join(
            # 이미지 경로를 Django 프로젝트 내 media 폴더로 변경
            'answer_image', 
            os.path.basename(output_path)
            )
        return django_path
        #
    #
    def _save_results(self, image_path, result_image, predictions):
        """결과 이미지를 저장"""
        os.makedirs(self.config['results_folder'], exist_ok=True) # 결과 폴더 생성
        output_path = os.path.join(self.config['results_folder'], os.path.basename(image_path)) # 결과 이미지 경로
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)) # 이미지 저장
        logging.info(f"이미지 분석 결과 저장:\n{output_path}")  
        #
        return output_path
        #
    #
#
def setup_warnings_and_logging():
    """ 경고 및 로깅 설정 """
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #
#
def load_config(config_path):
    """ 경로 설정 파일 로드 """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        #
    #
    # base_dir은 현재 파일의 부모 디렉토리로 설정
    base_dir = settings.BASE_DIR
    print(f'================{base_dir}====================')
    #
    # config 파일 내 상대 경로들을 절대 경로로 변환
    for key, value in config.items():
        print(f'================{key}====================')
        config[key] = os.path.join(base_dir, value)
        print(f'================{config[key]}====================')
        #
    #
    return config
    #
#
def setup_django_system(func):
    """설정 및 시스템 초기화를 처리하는 데코레이터"""
    def wrapper(request, image_path, *args, **kwargs):
        # 경고 및 로깅 설정
        setup_warnings_and_logging()
        
        # 설정 파일 로드
        config_path = settings.FACE_RECOGNITION_CONFIG
        print(f'================{config_path}====================')
        config = load_config(config_path)
        
        # 사용자가 요청에서 선택한 모델을 가져옴 (POST나 GET 파라미터로 전달 가능)
        selected_detectors = request.POST.getlist('detectors')  # 여러 탐지기 선택 가능
        selected_predictors = request.POST.getlist('predictors')  # 여러 예측기 선택 가능
        
        # 얼굴 탐지기 생성 - 사용자가 선택한 탐지기들을 설정
        detectors = []
        if 'dlib' in selected_detectors:
            detectors.append(DlibFaceDetector(config['dlib_model_path']))
        if 'yolo' in selected_detectors:
            detectors.append(YOLOFaceDetector(config['yolo_model_path']))
        if 'mtcnn' in selected_detectors:
            detectors.append(MTCNNFaceDetector())
        
        detector_manager = FaceDetectors(*detectors)
        
        # 얼굴 예측기 생성 - 사용자가 선택한 예측기들을 설정
        predictors = []
        if 'fairface' in selected_predictors:
            predictors.append(FairFacePredictor(config['fair_face_model_path']))
        
        predictor_manager = FacePredictors(*predictors)
        
        # 얼굴 인식 시스템 생성
        face_recognition_system = FaceRecognitionSystem(config, detector_manager, predictor_manager)
        
        # 타겟 얼굴 인코딩 로드
        with open(config['pickle_path'], 'rb') as f:
            target_encodings = np.array(pickle.load(f))
        
        # 함수 실행
        return func(request, image_path, face_recognition_system, target_encodings, *args, **kwargs)
    
    return wrapper
#
# =========================
# 메인 실행 모듈
# =========================
def main():
    """메인 함수: 시스템 설정 및 여러 이미지 처리"""
    # 경고 및 로깅 설정
    setup_warnings_and_logging()
    #
    # 설정 파일 로드
    config_path = os.path.join(Path(__file__).resolve().parent, 'config.json')
    config = load_config(config_path)
    #
    # 얼굴 탐지기 생성
    detector_manager = FaceDetectors(
        DlibFaceDetector(config['dlib_model_path']),
        YOLOFaceDetector(config['yolo_model_path']),
        MTCNNFaceDetector()
        )
    #
    # 얼굴 예측기 생성
    predictor_manager = FacePredictors(
        FairFacePredictor(config['fair_face_model_path'])
        )
    #
    # 얼굴 인식 시스템 생성
    face_recognition_system = FaceRecognitionSystem(config, detector_manager, predictor_manager)
    #
    # 타겟 얼굴 인코딩 로드
    with open(config['pickle_path'], 'rb') as f:
        target_encodings = np.array(pickle.load(f))
    #
    # 이미지 폴더에서 이미지 로드
    image_list = [f for f in os.listdir(config['image_folder']) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    #
    # 모든 이미지 처리
    for image in image_list:
        image_path = os.path.join(config['image_folder'], image)
        logging.info(f"이미지 처리 시작: {image_path}")
        output_path = face_recognition_system.process_image(image_path, target_encodings)
        logging.info(f"이미지 처리 완료: {output_path}")
        #
    #
#
if __name__ == "__main__":
    main()  
    #
#
