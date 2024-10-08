import os
from ast import Bytes
import base64

from fastapi import FastAPI,Body,Form,File, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageFilter
from io import BytesIO

from starlette.responses import JSONResponse
import ai_system
import cv2
import io
from ai import process_image, detect_president,compare_faces
fastapi_app = FastAPI()

#클라이언트에서 받은 이미지를 파일로저장 처리후에 삭제할지 결정
UPLOAD_AI_IMAGE = 'uploads_ai_image'
UPLOAD_SIMILARITY_IMAGE = 'uploads_similarity_image'
os.makedirs(UPLOAD_AI_IMAGE, exist_ok = True)
os.makedirs(UPLOAD_SIMILARITY_IMAGE, exist_ok = True)

@fastapi_app.post("/process_ai_image/")
async def process_ai_image(file: UploadFile = File(...)):


    file_location = os.path.join(UPLOAD_AI_IMAGE, file.filename)

    with open(file_location, "wb") as file_object:
        content = await file.read()
        file_object.write(content)

    result_image_path = detect_president(file_location)

    with open(result_image_path, "rb") as file_result:
        result_image =  file_result.read()

    if result_image :
        base64_image = base64.b64encode(result_image).decode('utf-8')
        print("64base image")

        result_data = {
            'message':"이 사진 속 인물은 도널드 트럼프(Donald Trump)입니다. 그는 미국의 제45대 대통령으로 2017년부터 2021년까지 재임했으며, 정치인이기 이전에는 부동산 개발업자이자 TV 방송인으로도 유명했습니다. 트럼프는 2016년 대통령 선거에서 공화당 후보로 출마해 승리했으며, 재임 중에는 ‘미국 우선주의’를 내세워 보호무역, 이민 제한, 세금 감면 등의 정책을 추진했습니다.",
            'base64_image':base64_image,
            'image_path': result_image_path
        }

        return result_data
    else:
        
        return "error발생"
    #이미지 경로로 읽은 이미지와와 결과 컨텐츠를 보내줌





@fastapi_app.post("/process_ai_image_two/")
async def process_ai_image_two(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    print("!!!called ai_two====2222222222")

    if file1 and file2 :
        try:

            file_location1 = os.path.join(UPLOAD_SIMILARITY_IMAGE, file1.filename)
            file_location2 = os.path.join(UPLOAD_SIMILARITY_IMAGE, file2.filename)

            print("file location 1", file_location1)
            print("file location 2", file_location2)
            with open(file_location1, "wb") as file_object1:
                content = await file1.read()
                file_object1.write(content)

            with open(file_location2, "wb") as file_object2:
                content = await file2.read()
                file_object2.write(content)


            similarity_percent = compare_faces(file_location1, file_location2)


            result1 = f"두 사진의 유사도는 {similarity_percent:.2f}입니다"
            return {'result':result1}

        except Exception as e:
            print("error server",e)
            result1 = "AI가 처리중에 에러가 발생하였습니다. 정확한 이미지를 올려주세요(1명 이미지)"
            return {'result':result1}

    return  {'result':"error 발생"}


if __name__=="__main__":
    import uvicorn
    uvicorn.run(fastapi_app,host="0.0.0.0",port=8007)

