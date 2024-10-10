"""
models/FaceEncoder.py

이 모듈은 FaceEncoder 클래스를 정의하며, 이미지에서 탐지된 얼굴 영역에 대한 인코딩을 생성하는 역할을 합니다.
"""

from ..core.config import PipelineStep
import face_recognition

class FaceEncoder(PipelineStep):
    """
    FaceEncoder 클래스는 이미지에서 탐지된 얼굴 영역에 대한 얼굴 인코딩을 생성하는 파이프라인 단계입니다.

    PipelineStep을 상속받아 구현되었으며, 데이터를 처리하는 `process` 메서드를 통해
    이미지에서 얼굴을 인코딩하여 데이터 객체에 추가하는 역할을 합니다.

    Attributes
    ----------
    None
    """

    def process(self, data):
        """
        데이터를 받아 얼굴 인코딩을 생성하고 데이터 객체에 저장합니다.

        `data` 객체는 이미지와 얼굴 영역 정보를 포함하고 있으며, 이를 기반으로
        face_recognition 라이브러리를 사용하여 얼굴 인코딩을 생성합니다.

        Parameters
        ----------
        data : Data
            파이프라인에서 공유되는 데이터 객체로, 'image_rgb'와 'predictions' 속성을 포함해야 합니다.
            'image_rgb'는 RGB 포맷의 이미지, 'predictions'는 검출된 얼굴 영역을 포함한 정보를 담고 있습니다.

        Returns
        -------
        Data
            얼굴 인코딩이 추가된 데이터 객체를 반환합니다.
            생성된 인코딩은 `data.encodings` 속성에 저장됩니다.
        """
        # 이미지와 얼굴 영역 정보를 데이터 객체에서 가져옴
        image_rgb = data.image_rgb
        faces = data.predictions['face_boxes']
        encodings = []

        # 각 얼굴 영역에 대해 인코딩을 생성
        for face in faces:
            x, y, x2, y2 = face  # 얼굴 영역 좌표를 얻음 (왼쪽, 위쪽, 오른쪽, 아래쪽)
            
            # face_recognition 라이브러리를 사용하여 얼굴 인코딩을 생성
            # 좌표는 (top, right, bottom, left) 순서로 전달해야 함
            encoding = face_recognition.face_encodings(image_rgb, [(y, x2, y2, x)])
            
            # 인코딩이 생성되었는지 확인하고, 리스트에 추가 (없을 경우 None을 추가)
            encodings.append(encoding[0] if encoding else None)

        # 인코딩 결과를 데이터 객체의 `encodings` 속성에 저장
        data.encodings = encodings
        
        # 수정된 데이터 객체를 반환
        return data
