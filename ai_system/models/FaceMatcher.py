"""
models/FaceMatcher.py

이 모듈은 TargetFaceMatcher 클래스를 정의하며,
탐지된 얼굴 인코딩을 사전에 저장된 타겟 얼굴 인코딩과 비교하여
일치 여부를 판별하는 역할을 합니다.
"""

from ..core.config import PipelineStep
import face_recognition

class TargetFaceMatcher(PipelineStep):
    """
    TargetFaceMatcher 클래스는 탐지된 얼굴의 인코딩을
    타겟 얼굴 인코딩과 비교하여 해당 얼굴이 타겟 인물인지 확인하는 클래스입니다.

    PipelineStep을 상속받아 구현되었으며, 파이프라인에서 얼굴 매칭 단계를 담당합니다.

    Attributes
    ----------
    target_encodings : list
        타겟 얼굴 인코딩의 리스트로, 비교 대상이 되는 얼굴의 인코딩입니다.
    """

    def __init__(self, target_encodings: list):
        """
        TargetFaceMatcher 클래스의 생성자.

        타겟 얼굴의 인코딩 리스트를 받아서 초기화합니다.

        Parameters
        ----------
        target_encodings : list
            타겟 얼굴의 인코딩 리스트입니다. 얼굴 인코딩은 사전에 저장된 값이어야 합니다.
        """
        self.target_encodings = target_encodings

    def process(self, data):
        """
        얼굴 인코딩을 받아 타겟 얼굴과의 매칭 여부를 판별하고, 결과를 저장합니다.

        Pipeline에서 `data` 객체를 받아, 탐지된 얼굴 인코딩과 사전에 저장된 타겟 얼굴 인코딩을
        비교하여 매칭 여부를 확인합니다. 매칭 결과는 `data.predictions['is_target']`에 저장됩니다.

        Parameters
        ----------
        data : Data
            파이프라인에서 공유되는 데이터 객체입니다. 이 객체는 'encodings' 속성을 포함해야 하며,
            해당 속성은 탐지된 얼굴 인코딩 리스트를 가지고 있어야 합니다.

        Returns
        -------
        Data
            얼굴 매칭 결과가 추가된 데이터 객체를 반환합니다.
        """
        encodings = data.encodings  # 데이터 객체에서 탐지된 얼굴 인코딩을 가져옴
        is_target_list = []  # 각 얼굴이 타겟 인물인지 여부를 저장할 리스트

        # 각 얼굴 인코딩에 대해 타겟 얼굴과의 매칭 여부 확인
        for encoding in encodings:
            if encoding is not None:
                # face_recognition 라이브러리를 사용하여 타겟 인코딩과 비교
                matches = face_recognition.compare_faces(
                    self.target_encodings,
                    encoding,
                    tolerance=0.3  # 허용 오차(tolerance)를 설정
                )
                # 하나라도 매칭되면 타겟 인물로 간주
                is_target = any(matches)
            else:
                is_target = False  # 인코딩이 없으면 매칭 불가

            is_target_list.append(is_target)

        # 매칭 결과를 데이터 객체에 저장
        data.predictions['is_target'] = is_target_list  # 매칭 여부 리스트를 predictions에 저장
        data.is_target_list = is_target_list  # 매칭 결과를 별도로 저장
        return data
