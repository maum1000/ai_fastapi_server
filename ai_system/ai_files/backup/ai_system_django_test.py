from pybo.ai_system.systems.face_detection_system_test_all import setup_system
#
@setup_system
def django_image_process(image_path, detectors, predictors, face_recognition_system, target_encodings):
    # 얼굴 인식 시스템을 이용해 이미지를 처리
    output_path = face_recognition_system.process_image(image_path, target_encodings)
    #
    return output_path