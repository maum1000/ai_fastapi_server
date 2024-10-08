# steps/save_results_step.py
from ..core.config import PipelineStep
import os
import cv2
import logging

class Saver(PipelineStep):

    def process(self, data):
        config = data.config
        image_path = data.image_path
        result_image = data.image_rgb
        # 결과 이미지를 저장할 경로 설정
        output_image_path = os.path.join(
            config['results_folder'], os.path.basename(image_path)
        )
        data.output_image_path = output_image_path
        os.makedirs(config['results_folder'], exist_ok=True)
        cv2.imwrite(output_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        logging.info(f"Saver : ai가 그린 이미지 저장: {output_image_path}")

        return data
