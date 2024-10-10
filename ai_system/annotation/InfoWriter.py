from ..core.config import PipelineStep

class InfoWriter(PipelineStep):
    """
    통계 정보를 이미지에 표시하는 파이프라인 단계입니다.

    이 클래스는 파이프라인 데이터를 사용하여 검출된 얼굴 수, 성별, 인종 등의 통계 정보를 이미지에 텍스트로 추가합니다.
    """

    def __init__(self, font_size=12):
        """
        InfoWriter 클래스를 초기화합니다.

        Args:
            font_size (int, optional): 텍스트를 표시할 때 사용할 기본 글꼴 크기. 기본값은 12입니다.
        """
        self.font_size = font_size

    def process(self, data):
        """
        이미지에 통계 정보를 텍스트로 추가합니다.

        Args:
            data (object): 파이프라인 데이터 객체로, 이미지와 통계 정보를 포함해야 합니다.
                           'image_rgb' 속성에는 이미지가 포함되어 있으며,
                           'predictions' 속성에는 통계 정보가 포함되어야 합니다.

        Returns:
            object: 통계 정보가 텍스트로 추가된 이미지가 포함된 데이터 객체.
                    'image_rgb' 속성이 업데이트됩니다.
        """

        # 이미지 처리 유틸리티 인스턴스 가져오기
        utils = data.image_utils

        # 원본 이미지 가져오기 ('image_rgb' 속성에서 이미지 데이터를 불러옴)
        image_rgb = data.image_rgb

        # 예측된 통계 정보 가져오기 ('predictions' 속성에서 'count' 정보)
        count = data.predictions['count']

        # 이미지 크기에 맞추어 글꼴 크기를 조정 (최소 self.font_size 이상)
        adaptive_font_size = max(self.font_size, int(image_rgb.shape[1] / 200))

        # 통계 정보를 텍스트로 변환 (인원 수, 성별 정보)
        info_text = (
            f"검출된 인원 수: {count['face_cnt']}명\n"
            f"남성: {count['male_cnt']}명\n"
            f"여성: {count['face_cnt'] - count['male_cnt']}명\n"
        )

        # 인종별 통계 정보를 텍스트로 변환 (인종별 인원 수 추가)
        race_info = "\n".join(
            [f"{race}: {count}명" for race, count in count['race_cnt'].items() if count > 0]
        )

        # 텍스트를 이미지에 추가 (이미지 크기에 맞춘 글꼴 크기 사용)
        image_rgb = utils.extend_image_with_text(
            image=image_rgb,
            text=info_text + race_info,  # 통합된 통계 정보 텍스트
            font_size=adaptive_font_size  # 조정된 글꼴 크기
        )

        # 텍스트가 추가된 이미지를 데이터 객체에 저장
        data.image_rgb = image_rgb

        # 처리된 데이터 객체 반환 (다음 파이프라인 단계로 전달)
        return data
