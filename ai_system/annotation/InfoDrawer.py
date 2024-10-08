# steps/draw_annotations_step.py

from ..core.config import PipelineStep
from ..core.utils import ImagePipeline

class InfoDrawer(PipelineStep):
    """
    얼굴 인식 결과(바운딩 박스 등)를 이미지에 표시하는 파이프라인 단계입니다.
    """
    def __init__(self, thickness=2):
        """
        InfoDrawer 클래스를 초기화합니다.

        Args:
            thickness (int): 박스의 두께를 지정합니다.
        """
        self.thickness = thickness

    def process(self, data):
        """
        이미지에 얼굴 박스와 텍스트 주석을 그려넣습니다.

        Args:
            data: 파이프라인 데이터 객체.

        Returns:
            data: 주석이 추가된 이미지가 포함된 데이터 객체.
        """
        config = data.config
        pipeline = ImagePipeline(config)
        image_rgb = data.image_rgb
        predictions = data.predictions
        face_boxes = predictions['face_boxes']
        is_target_list = predictions.get('is_target', [False] * len(face_boxes))  # 기본값 False로 채우기
        gender_list = predictions.get('gender', ['남성'] * len(face_boxes))  # 기본값 남성으로 채우기

        for index, face in enumerate(face_boxes):
            x1, y1, x2, y2 = face  # 얼굴 박스 좌표

            # 인덱스 범위를 벗어나지 않도록 gender 리스트의 값 가져오기
            gender = gender_list[index] if index < len(gender_list) else '남성'

            # 성별에 따라 박스 색상 결정 (남성: 파랑색, 여성: 빨간색)
            box_color = (50, 100, 255) if gender == '남성' else (255, 100, 50)

            # is_target 리스트의 값에 따라 텍스트 설정
            text = '가카!' if is_target_list[index] else ''
                
            # 이미지에 한글 텍스트 추가
            image_rgb = pipeline.draw_korean_text(
                image=image_rgb,
                text=text,
                position=(x1, y1),
                font_size=15,
                font_color=(0, 0, 0),
                background_color=box_color
            )

            # 얼굴 박스 그리기
            pipeline.draw_rectangle(image_rgb, (x1, y1, x2, y2), box_color, thickness=self.thickness)

        # 처리된 이미지를 데이터 객체에 저장
        data.image_rgb = image_rgb
        return data
