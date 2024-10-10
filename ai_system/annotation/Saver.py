from ..core.config import PipelineStep
import os
import cv2
import logging

class Saver(PipelineStep):
    """
    처리된 이미지를 파일로 저장하는 파이프라인 단계입니다.

    이 클래스는 AI가 처리한 이미지를 지정된 결과 폴더에 저장합니다.
    """

    def process(self, data):
        """
        이미지를 지정된 경로에 저장합니다.

        Args:
            data (object): 파이프라인 데이터 객체로, 'image_rgb' 속성에 처리된 이미지가 포함되어 있고,
                           'image_path' 속성에 원본 이미지 경로가 포함되어 있어야 합니다.
                           'config' 속성에 결과 폴더 경로가 포함된 설정 정보를 제공해야 합니다.

        Returns:
            object: 저장 경로가 포함된 데이터 객체. 'output_image_path' 속성이 추가됩니다.
        """
        # 데이터 객체에서 설정(config) 정보 가져오기
        config = data.config

        # 원본 이미지 경로와 처리된 이미지 가져오기
        image_path = data.image_path
        result_image = data.image_rgb

        # 결과 이미지를 저장할 경로 설정 (결과 폴더와 파일 이름)
        output_image_path = os.path.join(
            config['results_folder'],  # 결과 파일을 저장할 폴더
            os.path.basename(image_path)  # 원본 이미지 이름을 사용
        )

        # 데이터 객체에 결과 이미지 경로 저장
        data.output_image_path = output_image_path

        # 결과 폴더가 없는 경우 생성 (폴더가 이미 있으면 생략)
        os.makedirs(config['results_folder'], exist_ok=True)

        # 결과 이미지를 BGR 형식으로 변환한 후 파일로 저장
        cv2.imwrite(output_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

        # 저장된 이미지 경로를 로그로 출력
        logging.info(f"Saver : AI가 처리한 이미지를 저장했습니다: {output_image_path}")

        # 데이터 객체 반환 (다음 파이프라인 단계로 전달)
        return data
