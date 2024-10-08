# steps/encode_faces.py

"""
steps/encode_faces.py

이 모듈은 FaceEncoder 클래스를 정의하며, 이미지에서 탐지된 얼굴 영역에 대한 인코딩을 생성하는 역할을 합니다.
"""

from ..core.config import PipelineStep
import face_recognition

class FaceEncoder(PipelineStep):
    """
    FaceEncoder 클래스는 이미지에서 탐지된 얼굴 영역에 대한 얼굴 인코딩을 생성합니다.

    이 클래스는 PipelineStep을 상속받아 구현되었으며, 데이터 객체의 'faces' 속성에 있는
    얼굴 영역에 대해 face_recognition 라이브러리를 사용하여 얼굴 인코딩을 생성합니다.
    생성된 인코딩은 데이터 객체의 'encodings' 속성에 저장됩니다.
    """

    def process(self, data):
        """
        데이터를 받아 얼굴 인코딩을 생성하고 데이터 객체에 저장합니다.

        Args:
            data (Data): 파이프라인에서 공유되는 데이터 객체로,
                        'image_rgb'와 'faces' 속성을 포함해야 합니다.

        Returns:
            Data: 얼굴 인코딩이 추가된 데이터 객체를 반환합니다.
        """
        image_rgb = data.image_rgb
        faces = data.predictions['face_boxes']
        encodings = []
        # 각 얼굴 영역에 대해 인코딩 생성
        for face in faces:
            x, y, x2, y2 = face
            # face_recognition 라이브러리를 사용하여 얼굴 인코딩 생성
            # 좌표는 (top, right, bottom, left) 순서로 전달해야 합니다.
            encoding = face_recognition.face_encodings(image_rgb, [(y, x2, y2, x)])
            # 인코딩이 생성되었는지 확인하고 리스트에 추가
            encodings.append(encoding[0] if encoding else None)
        # 데이터 객체에 인코딩 결과 저장
        data.encodings = encodings
        return data
