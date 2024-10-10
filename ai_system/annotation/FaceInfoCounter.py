"""
steps/collect_statistics_step.py

FaceInfoCounter 클래스는 얼굴 인식 결과로부터 통계 정보를 수집하는
파이프라인 단계를 정의합니다.
"""
import logging
from ..core.config import PipelineStep


class FaceInfoCounter(PipelineStep):
    """
    얼굴 인식 결과로부터 통계 정보를 수집하는 파이프라인 단계입니다.

    Attributes:
        없음
    """

    def process(self, data):
        """
        데이터를 처리하여 얼굴 통계를 수집하고, 수집된 통계를 예측 결과에 추가합니다.

        Args:
            data: 파이프라인 데이터 객체.
                  'predictions' 속성에는 얼굴 인식 예측 정보가 포함되어 있어야 하며,
                  'is_target_list' 속성에는 대상 여부 리스트가 포함되어야 합니다.

        Returns:
            data: 통계 정보가 추가된 데이터 객체.
                  'predictions' 딕셔너리에 'count'라는 키로 통계 정보가 추가됩니다.
        """
        # 예측 결과와 대상 여부 리스트 가져오기
        predictions = data.predictions
        is_target_list = data.is_target_list

        # 대상 여부 리스트가 없을 경우 에러 메시지 출력 후 데이터 반환
        if is_target_list is None:
            logging.error("대상 여부 리스트가 없습니다.")
            return data

        # 통계를 저장할 딕셔너리 초기화
        count = {
            'face_cnt': 0,  # 총 얼굴 수
            'male_cnt': 0,  # 남성 얼굴 수
            'race_cnt': {   # 인종별 얼굴 수
                '백인': 0,
                '흑인': 0,
                '아시아': 0,
                '중동': 0
            }
        }

        # 검출된 얼굴의 수
        num_faces = len(predictions['face_boxes'])

        # 얼굴 수만큼 반복하여 각 예측 결과에 대해 통계 수집
        for idx in range(num_faces):
            count['face_cnt'] += 1  # 얼굴 수 증가

            # 남성인 경우 남성 수 증가
            if predictions['gender'][idx] == '남성':
                count['male_cnt'] += 1

            # 각 인종별로 카운트 증가
            race = predictions['race'][idx]
            if race in count['race_cnt']:
                count['race_cnt'][race] += 1

        # 수집된 통계를 예측 결과에 추가
        data.predictions.update({
            'count': count  # 'count' 키에 통계 정보 추가
        })

        # 로그에 최종 예측 결과와 통계 정보 출력
        for key, value in data.predictions.items():
            logging.info(f"FaceInfoCounter: \n{key}: {value}")

        return data
