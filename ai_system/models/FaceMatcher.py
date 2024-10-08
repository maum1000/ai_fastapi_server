# steps/target_face_matcher.py

"""
steps/target_face_matcher.py

이 모듈은 TargetFaceMatcher 클래스를 정의하며,
탐지된 얼굴 인코딩을 사전에 저장된 타겟 얼굴 인코딩과 비교하여
일치 여부를 판별하는 역할을 합니다.
"""

from ..core.config import PipelineStep
import face_recognition

class TargetFaceMatcher(PipelineStep):
    """
    TargetFaceMatcher 클래스는 탐지된 얼굴의 인코딩을
    타겟 얼굴 인코딩과 비교하여 해당 얼굴이 타겟 인물인지 확인합니다.

    이 클래스는 PipelineStep을 상속받아 구현되었으며,
    파이프라인에서 얼굴 매칭 단계를 담당합니다.
    """

    def __init__(self, target_encodings):
        """
        TargetFaceMatcher를 초기화합니다.

        Args:
            target_encodings (list): 타겟 얼굴의 인코딩 리스트입니다.
        """
        self.target_encodings = target_encodings

    def process(self, data):
        """
        데이터를 받아 탐지된 얼굴이 타겟 인물인지 여부를 판별하고 결과를 저장합니다.

        Args:
            data (Data): 파이프라인에서 공유되는 데이터 객체로,
                        'encodings' 속성을 포함해야 합니다.

        Returns:
            Data: 얼굴 매칭 결과가 추가된 데이터 객체를 반환합니다.
        """
        encodings = data.encodings
        is_target_list = []

        # 각 얼굴 인코딩에 대해 타겟 얼굴과의 매칭 여부 확인
        for encoding in encodings:
            if encoding is not None:
                # 타겟 얼굴 인코딩 리스트와 비교하여 매칭 여부 확인
                matches = face_recognition.compare_faces(
                    self.target_encodings,
                    encoding,
                    tolerance=0.3
                )
                is_target = any(matches)
            else:
                is_target = False  # 인코딩이 없으면 매칭 불가

            is_target_list.append(is_target)
        data.predictions['is_target'] = is_target_list
        # 매칭 결과를 데이터 객체에 저장
        data.is_target_list = is_target_list
        return data
