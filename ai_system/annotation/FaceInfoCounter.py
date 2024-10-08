# steps/collect_statistics_step.py

import logging
from ..core.config import PipelineStep

class FaceInfoCounter(PipelineStep):
    """
    얼굴 인식 결과로부터 통계를 수집하는 파이프라인 단계입니다.
    """
    def process(self, data):
        """
        데이터를 처리하여 통계 정보를 수집하고, 예측 결과에 포함시킵니다.

        Args:
            data: 파이프라인 데이터 객체. 예측 결과와 대상 여부 리스트를 포함해야 합니다.

        Vars:
            predictions = data.predictions : 예측 정보 Dict
            
            현제 data.predictions 구조
            data.predictions = {
                'face_box': [(x1, y1, x2, y2), (x1, y1, x2, y2), ...],
                'race': ['백인', '아시아', ...],
                'gender': ['남성', '여성', ...],
                'age': ['30대', '20대', ...]
            }
            
            is_target_list = data.is_target_list : 대상 여부 리스트

        Returns:
            data: 통계 정보가 추가된 데이터 객체.
        """
        # 예측 결과와 대상 여부 리스트 가져오기
        predictions = data.predictions
        is_target_list = data.is_target_list
        
        if is_target_list is None:
            logging.error("대상 여부 리스트가 없습니다.")
            return data

        # 통계 정보를 저장할 딕셔너리 초기화
        count = {
            'face_cnt': 0,
            'male_cnt': 0,
            'race_cnt': {'백인': 0, '흑인': 0, '아시아': 0, '중동': 0}
        }

        # 얼굴 수만큼 반복하여 예측 결과와 통계 수집
        num_faces = len(predictions['face_boxes'])
        for idx in range(num_faces):
            # 통계 정보 업데이트
            count['face_cnt'] += 1
            if predictions['gender'][idx] == '남성':
                count['male_cnt'] += 1
            race = predictions['race'][idx]
            if race in count['race_cnt']:
                count['race_cnt'][race] += 1

        # 데이터 객체에 최종 예측 결과와 통계 정보 추가
        data.predictions.update({
            'count': count
        })

        for key, value in data.predictions.items():
            logging.info(f"FaceInfoCounter : \n{key}: {value}")
        return data
