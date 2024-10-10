"""
steps/image_resize_step.py

이 모듈은 ImageResizer 클래스를 정의하며, 이미지를 리사이즈하고 패딩을 추가하는 역할을 합니다.
"""

from ..core.config import PipelineStep


class ImageResizer(PipelineStep):
    """
    이미지 리사이즈 및 패딩 추가 클래스입니다.

    이 클래스는 PipelineStep을 상속받아 구현되었으며, 이미지를 주어진 크기로 리사이즈하고,
    종횡비(가로세로 비율)를 유지하기 위해 필요한 경우 패딩을 추가합니다.
    파이프라인의 이미지 전처리 단계에서 사용됩니다.
    """

    def __init__(self, target_size=512):
        """
        ImageResizer 클래스의 인스턴스를 초기화합니다.

        Args:
            target_size (int, optional): 리사이즈할 이미지의 목표 크기입니다. 
                                         기본값은 512이며, 이 값은 이미지의 너비 또는 높이가 됩니다.
        """
        self.target_size = target_size

    def process(self, data):
        """
        파이프라인 데이터 객체를 받아 이미지를 리사이즈하고 패딩을 추가합니다.

        Args:
            data (object): 파이프라인에서 공유되는 데이터 객체입니다.
                           'image_rgb' 속성에 이미지가 포함되어 있으며,
                           'config' 속성에 설정 정보가 포함되어야 합니다.

        Returns:
            object: 리사이즈 및 패딩이 적용된 이미지를 포함하는 데이터 객체.
                    'image_rgb', 'scale', 'top', 'left' 속성이 업데이트됩니다.
        """
        # 이미지 처리 유틸리티 인스턴스 가져오기
        utils = data.image_utils
        
        # 원본 이미지 가져오기 ('image_rgb' 속성에서 이미지 데이터를 불러옴)
        image_rgb = data.image_rgb
        
        # 이미지 리사이즈 및 패딩 추가 작업 수행
        resized_image_rgb, scale, top, left = utils.resize_image_with_padding(
            image=image_rgb,             # 원본 이미지
            target_size=self.target_size, # 목표 크기
            padding_color=(0, 0, 0)      # 패딩 색상 (검은색)
        )
        
        # 처리된 이미지를 데이터 객체에 저장 (기존 이미지 대체)
        data.image_rgb = resized_image_rgb
        # 리사이즈 과정에서 계산된 비율과 패딩 정보도 추가
        data.scale = scale
        data.top = top
        data.left = left
        
        # 업데이트된 데이터를 반환하여 파이프라인의 다음 단계로 전달
        return data
