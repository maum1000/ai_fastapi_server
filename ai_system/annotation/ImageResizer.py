# steps/image_resize_step.py

"""
steps/image_resize_step.py

이 모듈은 ImageResizer 클래스를 정의하며, 이미지를 리사이즈하고 패딩을 추가하는 역할을 합니다.
"""

from ..core.config import PipelineStep
from ..core.utils import ImagePipeline

class ImageResizer(PipelineStep):
    """
    이미지 리사이즈 및 패딩 추가 클래스입니다.

    이 클래스는 PipelineStep을 상속받아 구현되었으며, 이미지를 원하는 크기로 리사이즈하고
    종횡비를 유지하기 위해 필요한 경우 패딩을 추가합니다. 파이프라인에서 이미지 전처리 단계로 사용됩니다.
    """

    def __init__(self, target_size=512):
        """
        ImageResizer를 초기화합니다.

        Args:
            target_size (int, optional): 리사이즈할 목표 크기입니다. 기본값은 512입니다.
        """
        self.target_size = target_size

    def process(self, data):
        """
        데이터를 받아 이미지를 리사이즈하고 패딩을 추가합니다.

        Args:
            data (Data): 파이프라인에서 공유되는 데이터 객체로, 'image_rgb'와 'config' 속성을 포함해야 합니다.

        Returns:
            Data: 이미지가 리사이즈되고 'image_rgb', 'scale', 'top', 'left' 속성이 업데이트된 데이터 객체를 반환합니다.
        """
        # 데이터 객체에서 설정 정보 가져오기
        config = data.config
        # ImagePipeline 인스턴스 생성
        pipeline = ImagePipeline(config)
        # 원본 이미지 가져오기
        image_rgb = data.image_rgb
        # 이미지 리사이즈 및 패딩 추가
        resized_image_rgb, scale, top, left = pipeline.resize_image_with_padding(
            image=image_rgb, 
            target_size=self.target_size,
            padding_color=(0, 0, 0)
        )
        # 데이터 객체에 업데이트된 정보 저장
        # data.resized_image_rgb = resized_image_rgb
        data.image_rgb = resized_image_rgb
        data.scale = scale
        data.top = top
        data.left = left
        # 처리된 데이터를 반환하여 다음 단계로 전달
        return data
