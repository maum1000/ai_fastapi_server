import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import io
import piexif
import os
import logging

"""
ai_system/steps.py

이 모듈은 ImagePipeline 클래스를 포함하고 있으며, 이미지 처리 파이프라인에서 사용되는 다양한 이미지 처리 유틸리티를 제공합니다.
ImagePipeline 클래스는 이미지 리사이징, 텍스트 추가, 이미지 확장, 사각형 그리기, 메타데이터 추가 등의 기능을 제공합니다.
"""


class ImageUtils:
    """
    이미지 처리 파이프라인에서 사용되는 유틸리티 메서드를 제공하는 클래스입니다.

    Methods
    -------
    resize_image_with_padding(image: np.ndarray, target_size: int, padding_color: tuple) -> tuple:
        이미지의 종횡비를 유지하면서 패딩을 추가하여 리사이즈합니다.

    draw_korean_text(image: np.ndarray, text: str, position: tuple, font_size: int, font_color: tuple, background_color: tuple) -> np.ndarray:
        이미지에 한글 텍스트를 그립니다.

    extend_image_with_text(image: np.ndarray, text: str, font_size: int, font_color: tuple, background_color: tuple) -> np.ndarray:
        이미지 상단에 텍스트를 추가하여 이미지를 확장합니다.

    draw_rectangle(image: np.ndarray, coordinates: tuple, color: tuple, thickness: int) -> None:
        이미지에 사각형을 그립니다.

    copy_image_and_add_metadata(image_path: str, output_folder: str, data: dict) -> None:
        이미지를 복사하고 메타데이터를 추가합니다.
    """

    def __init__(self, config: dict):
        """
        ImagePipeline을 초기화합니다.

        Parameters
        ----------
        config : dict
            'font_path' 등의 설정이 포함된 구성 딕셔너리입니다.
        """
        self.config = config
        self.font_path = config.get('font_path', None)
        if not self.font_path:
            logging.warning("설정에서 폰트 경로가 지정되지 않았습니다.")

    def resize_image_with_padding(self, image: np.ndarray, target_size: int, padding_color: tuple = (0, 0, 0)) -> tuple:
        """
        이미지의 종횡비를 유지하면서 패딩을 추가하여 원하는 크기로 리사이즈합니다.

        Parameters
        ----------
        image : np.ndarray
            리사이즈할 입력 이미지입니다.
        target_size : int
            출력 이미지의 원하는 크기(폭과 높이)입니다.
        padding_color : tuple, optional
            패딩 영역에 사용할 RGB 색상 값입니다. 기본값은 검정색 (0, 0, 0)입니다.

        Returns
        -------
        tuple
            - new_img (np.ndarray): 패딩이 추가된 리사이즈된 이미지.
            - scale (float): 이미지에 적용된 스케일 팩터.
            - top (int): 위쪽에 추가된 패딩의 픽셀 수.
            - left (int): 왼쪽에 추가된 패딩의 픽셀 수.
        """
        # 이미지에 적용할 스케일 팩터 계산
        scale = self._calculate_scale(image.shape, target_size)
        # 스케일 팩터를 사용하여 이미지 리사이즈
        resized_img = self._resize_image(image, scale)
        # 리사이즈된 이미지에 패딩 추가
        new_img, top, left = self._add_padding(resized_img, target_size, padding_color)
        return new_img, scale, top, left

    def draw_korean_text(
        self,
        image: np.ndarray,
        text: str,
        position: tuple,
        font_size: int,
        font_color: tuple = (255, 255, 255),
        background_color: tuple = (0, 0, 0),
    ) -> np.ndarray:
        """
        이미지의 지정된 위치에 한글 텍스트를 그립니다.

        Parameters
        ----------
        image : np.ndarray
            텍스트를 그릴 이미지입니다.
        text : str
            그릴 텍스트입니다.
        position : tuple
            텍스트를 배치할 (x, y) 좌표입니다.
        font_size : int
            텍스트의 폰트 크기입니다.
        font_color : tuple, optional
            텍스트의 RGB 색상입니다. 기본값은 흰색 (255, 255, 255)입니다.
        background_color : tuple, optional
            텍스트 배경의 RGB 색상입니다. 기본값은 검정색 (0, 0, 0)입니다.

        Returns
        -------
        np.ndarray
            텍스트가 그려진 이미지입니다.
        """
        if not text or not self.font_path:
            return image

        # 폰트 로드
        font = ImageFont.truetype(self.font_path, font_size)
        # 텍스트의 크기 측정
        text_size = self._measure_text_size(text, font)
        # 텍스트 배경 박스의 좌표 계산
        box_coords = self._calculate_text_box(position, text_size)
        # 텍스트를 그리기 위해 필요한 경우 이미지 확장
        image = self._extend_image_if_needed(
            image, (box_coords[2], box_coords[3]), background_color
        )
        # 이미지를 PIL 형식으로 변환
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        # 텍스트 배경 사각형 그리기
        draw.rectangle(box_coords, fill=background_color)
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=font_color)
        return np.array(image_pil)

    def extend_image_with_text(
        self,
        image: np.ndarray,
        text: str,
        font_size: int,
        font_color: tuple = (255, 255, 255),
        background_color: tuple = (0, 0, 0),
    ) -> np.ndarray:
        """
        이미지 상단에 텍스트를 추가하여 이미지를 확장합니다.

        Parameters
        ----------
        image : np.ndarray
            확장할 원본 이미지입니다.
        text : str
            상단에 추가할 텍스트입니다.
        font_size : int
            텍스트의 폰트 크기입니다.
        font_color : tuple, optional
            텍스트의 RGB 색상입니다. 기본값은 흰색 (255, 255, 255)입니다.
        background_color : tuple, optional
            확장된 영역의 배경 RGB 색상입니다. 기본값은 검정색 (0, 0, 0)입니다.

        Returns
        -------
        np.ndarray
            확장되고 텍스트가 추가된 이미지입니다.
        """
        if not self.font_path:
            logging.warning("폰트 경로가 지정되지 않았습니다. 텍스트 확장을 건너뜁니다.")
            return image

        # 텍스트를 표시하기 위해 필요한 추가 높이 계산
        extra_height = self._calculate_total_text_height(text, font_size)
        # 추가 높이가 있는 확장된 이미지 생성
        extended_image = self._create_extended_image(image, extra_height, background_color)
        # 폰트 로드
        font = ImageFont.truetype(self.font_path, font_size)
        # 이미지를 PIL 형식으로 변환
        image_pil = Image.fromarray(extended_image)
        draw = ImageDraw.Draw(image_pil)
        # 상단에 텍스트 그리기
        draw.text((10, 10), text, font=font, fill=font_color)
        return np.array(image_pil)

    def draw_rectangle(self, image: np.ndarray, coordinates: tuple, color: tuple, thickness: int) -> None:
        """
        이미지에 사각형을 그립니다.

        Parameters
        ----------
        image : np.ndarray
            사각형을 그릴 이미지입니다.
        coordinates : tuple
            사각형의 (x1, y1, x2, y2) 좌표입니다.
        color : tuple
            사각형 테두리의 RGB 색상입니다.
        thickness : int
            사각형 테두리의 두께(픽셀)입니다.

        Returns
        -------
        None
        """
        x1, y1, x2, y2 = coordinates
        # 사각형 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def copy_image_and_add_metadata(self, image_path: str, output_folder: str, data: dict) -> None:
        """
        이미지를 지정된 폴더로 복사하고 메타데이터를 추가합니다.

        Parameters
        ----------
        image_path : str
            원본 이미지의 경로입니다.
        output_folder : str
            이미지를 복사할 폴더입니다.
        data : dict
            메타데이터 정보가 포함된 데이터입니다.

        Returns
        -------
        None
        """
        # 이미지를 출력 폴더로 복사
        copied_image_path = self._copy_image_to_folder(image_path, output_folder)
        # 복사된 이미지에 메타데이터 추가
        self._add_metadata_to_image(copied_image_path, data)
        logging.info(f"메타데이터와 함께 이미지 저장됨: {copied_image_path}")

    # Private helper methods (사용자에게 노출되지 않는 내부 함수)
    def _calculate_scale(self, image_shape: tuple, target_size: int) -> float:
        """
        이미지를 목표 크기로 리사이즈하기 위한 스케일 팩터를 계산합니다.

        Parameters
        ----------
        image_shape : tuple
            이미지의 형태 (높이, 폭, 채널 수)입니다.
        target_size : int
            출력 이미지의 원하는 크기(폭과 높이)입니다.

        Returns
        -------
        float
            이미지를 리사이즈하기 위한 스케일 팩터입니다.
        """
        h, w = image_shape[:2]
        return target_size / max(h, w)

    def _resize_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """
        주어진 스케일 팩터를 사용하여 이미지를 리사이즈합니다.

        Parameters
        ----------
        image : np.ndarray
            리사이즈할 이미지입니다.
        scale : float
            이미지를 리사이즈할 스케일 팩터입니다.

        Returns
        -------
        np.ndarray
            리사이즈된 이미지입니다.
        """
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        # 이미지 리사이즈
        return cv2.resize(image, (new_w, new_h))

    def _add_padding(self, image: np.ndarray, target_size: int, padding_color: tuple = (0, 0, 0)) -> tuple:
        """
        이미지를 목표 크기로 만들기 위해 패딩을 추가합니다.

        Parameters
        ----------
        image : np.ndarray
            패딩을 추가할 이미지입니다.
        target_size : int
            출력 이미지의 원하는 크기(폭과 높이)입니다.
        padding_color : tuple, optional
            패딩 영역에 사용할 RGB 색상입니다.

        Returns
        -------
        tuple
            - padded_img (np.ndarray): 패딩이 추가된 이미지.
            - top (int): 위쪽에 추가된 패딩의 픽셀 수.
            - left (int): 왼쪽에 추가된 패딩의 픽셀 수.
        """
        delta_w = target_size - image.shape[1]
        delta_h = target_size - image.shape[0]
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left
        # 이미지에 패딩 추가
        padded_img = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
        )
        return padded_img, top, left

    def _measure_text_size(self, text: str, font: ImageFont.FreeTypeFont) -> tuple:
        """
        지정된 폰트로 텍스트를 그렸을 때의 크기(폭과 높이)를 측정합니다.

        Parameters
        ----------
        text : str
            측정할 텍스트입니다.
        font : PIL.ImageFont.FreeTypeFont
            텍스트를 렌더링하는 데 사용할 폰트입니다.

        Returns
        -------
        tuple
            - text_width (int): 텍스트의 폭(픽셀).
            - text_height (int): 텍스트의 높이(픽셀).
        """
        # 텍스트 크기 측정을 위한 더미 이미지 생성
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        # 텍스트의 바운딩 박스 얻기
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        return text_width, text_height

    def _calculate_text_box(self, position: tuple, text_size: tuple, padding: int = 5) -> tuple:
        """
        텍스트 배경 박스의 좌표를 계산합니다.

        Parameters
        ----------
        position : tuple
            텍스트 시작 위치의 (x, y) 좌표입니다.
        text_size : tuple
            텍스트의 (폭, 높이)입니다.
        padding : int, optional
            텍스트 주변에 추가할 패딩의 크기입니다.

        Returns
        -------
        tuple
            텍스트 배경 박스의 좌표 (왼쪽, 위쪽, 오른쪽, 아래쪽)입니다.
        """
        x, y = position
        text_width, text_height = text_size
        return (
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding,
        )

    def _extend_image_if_needed(self, image: np.ndarray, new_size: tuple, background_color: tuple = (0, 0, 0)) -> np.ndarray:
        """
        필요한 경우 이미지를 확장하여 새로운 크기에 맞춥니다.

        Parameters
        ----------
        image : np.ndarray
            원본 이미지입니다.
        new_size : tuple
            원하는 (폭, 높이) 크기입니다.
        background_color : tuple, optional
            확장된 영역에 사용할 RGB 색상입니다.

        Returns
        -------
        np.ndarray
            확장된 이미지입니다.
        """
        new_width, new_height = new_size
        height, width = image.shape[:2]
        if new_width > width or new_height > height:
            # 새로운 크기와 배경 색상을 가진 이미지 생성
            extended_img = np.full(
                (max(height, new_height), max(width, new_width), 3),
                background_color,
                dtype=np.uint8,
            )
            # 원본 이미지를 확장된 이미지에 배치
            extended_img[:height, :width] = image
            return extended_img
        return image

    def _calculate_total_text_height(self, text: str, font_size: int) -> int:
        """
        줄 간격과 패딩을 포함하여 텍스트를 표시하는 데 필요한 전체 높이를 계산합니다.

        Parameters
        ----------
        text : str
            표시할 텍스트입니다. 여러 줄을 포함할 수 있습니다.
        font_size : int
            텍스트의 폰트 크기입니다.

        Returns
        -------
        int
            텍스트를 표시하는 데 필요한 총 높이(픽셀).
        """
        lines = text.split('\n')
        line_height = font_size * 1.5  # 줄 간격을 포함한 줄 높이 추정
        return int(line_height * len(lines) + 20)  # 패딩 포함

    def _create_extended_image(self, image: np.ndarray, extra_height: int, background_color: tuple = (0, 0, 0)) -> np.ndarray:
        """
        상단에 추가 공간이 있는 새로운 이미지를 생성하고 원본 이미지를 아래에 배치합니다.

        Parameters
        ----------
        image : np.ndarray
            원본 이미지입니다.
        extra_height : int
            상단에 추가할 높이입니다.
        background_color : tuple, optional
            추가 공간에 사용할 RGB 색상입니다.

        Returns
        -------
        np.ndarray
            상단에 추가 공간이 있는 확장된 이미지입니다.
        """
        height, width = image.shape[:2]
        # 추가 높이를 가진 새로운 이미지 생성
        extended_img = np.full(
            (height + extra_height, width, 3), background_color, dtype=np.uint8
        )
        # 원본 이미지를 아래쪽에 배치
        extended_img[extra_height:, :] = image
        return extended_img

    def _copy_image_to_folder(self, image_path: str, output_folder: str) -> str:
        """
        이미지를 지정된 출력 폴더로 복사합니다.

        Parameters
        ----------
        image_path : str
            원본 이미지의 경로입니다.
        output_folder : str
            출력 폴더의 경로입니다.

        Returns
        -------
        str
            출력 폴더에 있는 복사된 이미지의 경로입니다.
        """
        # 출력 폴더가 존재하는지 확인하고 없으면 생성
        os.makedirs(output_folder, exist_ok=True)
        # 이미지 복사
        shutil.copy(image_path, output_folder)
        # 복사된 이미지의 경로 반환
        return os.path.join(output_folder, os.path.basename(image_path))

    def _add_metadata_to_image(self, image_path: str, data: dict) -> None:
        """
        이미지 파일에 메타데이터를 추가합니다.

        Parameters
        ----------
        image_path : str
            이미지 파일의 경로입니다.
        data : dict
            추가할 메타데이터 정보가 포함된 데이터입니다.

        Returns
        -------
        None
        """
        image = data.image_rgb
        with Image.open(image_path) as image:
            # 이미지가 RGB 모드인지 확인
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            meta_im = image.copy()
            thumb_im = image.copy()
            # EXIF 데이터에 사용할 썸네일 생성
            thumb_io = io.BytesIO()
            thumb_im.thumbnail((50, 50), Image.Resampling.LANCZOS)
            thumb_im.save(thumb_io, "jpeg")
            thumbnail = thumb_io.getvalue()

            # EXIF 데이터 정의
            zeroth_ifd = {
                piexif.ImageIFD.Make: u"oldcamera",
                piexif.ImageIFD.XResolution: (96, 1),
                piexif.ImageIFD.YResolution: (96, 1),
                piexif.ImageIFD.Software: u"piexif",
                piexif.ImageIFD.Artist: u"0!code",
            }

            exif_ifd = {
                piexif.ExifIFD.DateTimeOriginal: u"2099:09:29 10:10:10",
                piexif.ExifIFD.LensMake: u"LensMake",
                piexif.ExifIFD.Sharpness: 65535,
                piexif.ExifIFD.LensSpecification: (
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                ),
            }

            gps_ifd = {
                piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
                piexif.GPSIFD.GPSAltitudeRef: 1,
                piexif.GPSIFD.GPSDateStamp: u"1999:99:99 99:99:99",
            }

            first_ifd = {
                piexif.ImageIFD.Make: u"oldcamera",
                piexif.ImageIFD.XResolution: (40, 1),
                piexif.ImageIFD.YResolution: (40, 1),
                piexif.ImageIFD.Software: u"piexif",
            }

            exif_dict = {
                "0th": zeroth_ifd,
                "Exif": exif_ifd,
                "GPS": gps_ifd,
                "1st": first_ifd,
                "thumbnail": thumbnail,
            }
            # EXIF 데이터를 바이트로 덤프
            exif_bytes = piexif.dump(exif_dict)
            # EXIF 데이터를 포함하여 이미지 저장
            meta_im.save(image_path, exif=exif_bytes)

            image.save(image_path, exif=exif_bytes)
