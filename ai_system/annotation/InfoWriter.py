# steps/add_statistics_step.py
from ..core.config import PipelineStep
from ..core.utils import ImagePipeline

class InfoWriter(PipelineStep):
    def __init__(self, font_size=12):
        self.font_size = font_size

    def process(self, data):
        config = data.config
        pipeline = ImagePipeline(config)
        image_rgb = data.image_rgb
        count = data.predictions['count']
        adaptive_font_size = max(self.font_size, int(image_rgb.shape[1] / 200))

        info_text = (
            f"검출된 인원 수: {count['face_cnt']}명\n"
            f"남성: {count['male_cnt']}명\n"
            f"여성: {count['face_cnt'] - count['male_cnt']}명\n"
        )
        race_info = "\n".join(
            [f"{race}: {count}명" for race, count in count['race_cnt'].items() if count > 0]
        )

        image_rgb = pipeline.extend_image_with_text(
            image=image_rgb,
            text=info_text + race_info,
            font_size=adaptive_font_size
        )
        data.image_rgb = image_rgb
        return data
