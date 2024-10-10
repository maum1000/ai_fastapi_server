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


president_data= {'존슨':"이 사진 속 인물은 린든 B. 존슨(Lyndon B. Johnson)입니다. 린든 B. 존슨은 미국의 36대 대통령으로, 1963년부터 1969년까지 재임했습니다. 그는 대사회 프로그램과 민권법을 통해 사회 복지와 인권을 강화했으나, 베트남 전쟁으로 인해 큰 논란에 휘말렸습니다. 그의 정책은 오늘날에도 여전히 중요한 논의의 주제가 되고 있습니다",
                 "닉슨":"이 사진 속 인물은 리처트 닉슨(Richard Nixon)입니다. 리처드 닉슨은 미국의 37대 대통령으로, 1969년부터 1974년까지 재임했습니다. 그는 중국과의 외교 관계 정상화와 워터게이트 스캔들로 인해 사임하는 등 극명한 대조를 이루는 정치 경력을 가졌습니다. 또한, 환경 보호법과 같은 중요한 국내 정책도 추진했습니다.",
                 "포드":"이 사진 속 인물은 제럴드 포드(Gerald Ford)입니다. 제럴드 포드는 미국의 38대 대통령으로, 1974년부터 1977년까지 재임했습니다. 그는 리처드 닉슨의 사임 후 부통령으로 취임했으며, 경제적 어려움과 높은 인플레이션에 대응하기 위해 다양한 캠페인을 추진했습니다. 닉슨을 사면한 결정은 큰 논란을 일으켰고, 그의 재임 기간은 정치적으로 힘든 시기로 기억됩니다.",
                 "카터":"이 사진 속 인물은 지미 카터(Jimmy Carter)입니다.지미 카터는 미국의 39대 대통령으로, 1977년부터 1981년까지 재임했습니다. 그는 인권을 강조하고 에너지 위기 대응에 노력했으며, 캠프 데이비드 협정을 통해 중동 평화에 기여했습니다. 퇴임 후에도 인권과 사회 봉사 활동으로 많은 영향을 미쳤습니다.",
                 "레이건":"이 사진 속 인물은 로널드 레이건(Ronald Reagan)입니다.로널드 레이건은 미국의 40대 대통령으로, 1981년부터 1989년까지 재임했습니다. 그는 공급 측 경제학을 통해 경제 성장과 세금 감면을 추진하고, 냉전 종식에 기여했습니다. 또한, 군사력 증강과 사회 복지 프로그램 축소를 통해 보수주의 정책을 강화했습니다.",
                 "아빠 부시":"이 사진 속 인물은 조지 H.W. 부시((George H.W. Bush)입니다. 조지 H.W. 부시는 미국의 41대 대통령으로, 1989년부터 1993년까지 재임했습니다. 그는 냉전 종식과 걸프 전쟁에서의 군사 작전으로 국제 정치에서 중요한 역할을 했으나, 재임 중 경제 침체와 세금 인상 문제로 비판을 받았습니다. 또한, 환경 보호와 장애인 권리 증진을 위한 법안을 추진하며 사회 정책에도 기여했습니다.",
                 "클린턴":"이 사진 속 인물은 빌 클린턴(Bill Clinton)입니다빌 클린턴은 미국의 42대 대통령으로, 1993년부터 2001년까지 재임했습니다. 그는 재임 중 경제 성장을 이끌고 여러 사회 정책을 추진했으나, 모니카 르윈스키 스캔들로 인해 탄핵 절차를 겪었습니다. 클린턴 대통령은 경제적 성과와 정치적 논란이 얽힌 복잡한 평가를 받고 있습니다." ,
                 "아들 부시":"이 사진 속 인물은 조지 W. 부시(George W. Bush)입니다,조지 W. 부시는 미국의 43대 대통령으로, 2001년부터 2009년까지 재임했습니다. 그는 9/11 테러 이후 아프가니스탄과 이라크 전쟁을 주도하며 테러와의 전쟁을 선언했으나, 2008년 금융 위기로 인해 경제 정책에 대한 비판을 받았습니다. 교육 개혁과 보건 정책에도 관심을 기울였지만, 그의 재임 기간은 전쟁과 경제 문제로 논란이 많았습니다.",
                 '오바마': "이 사진 속 인물은 오바마 (Barack Obama)입니다.오바마는 미국의 44대 대통령으로, 2009년부터 2017년까지 재임했습니다. 그는 미국 역사상 첫 아프리카계 미국인 대통령이며, 재임 중 건강보험 개혁, 재정위기 대응, 외교 정책 변화 등의 여러 중요한 정책을 추진했습니다. 퇴임 후에도 사회적 이슈에 대한 목소리를 내고 있으며, 많은 사람들에게 영감을 주고 있습니다. ",
                 "트럼프":"이 사진 속 인물은 도널드 트럼프(Donald Trump)입니다. 그는 미국의 제45대 대통령으로 2017년부터 2021년까지 재임했으며, 정치인이기 이전에는 부동산 개발업자이자 TV 방송인으로도 유명했습니다. 트럼프는 2016년 대통령 선거에서 공화당 후보로 출마해 승리했으며, 재임 중에는 ‘미국 우선주의’를 내세워 보호무역, 이민 제한, 세금 감면 등의 정책을 추진했습니다.",
                 "바이든":"이 사진 속 인물은 조 바이든(Joe Biden)입니다.조 바이든은 미국의 46대 대통령으로, 2021년 1월 20일부터 재임 중입니다. 그는 COVID-19 대응과 경제 회복을 위한 법안을 추진하며 기후 변화 및 인종 평등과 같은 진보적 사회 정책을 지향하고 있습니다. 또한, 동맹국과의 관계를 강화하고 중국과의 경쟁에서 미국의 입장을 확립하려고 노력하고 있습니다."

                }
@fastapi_app.post("/process_ai_image/")
async def process_ai_image(file: UploadFile = File(...)):


    file_location = os.path.join(UPLOAD_AI_IMAGE, file.filename)

    with open(file_location, "wb") as file_object:
        content = await file.read()
        file_object.write(content)

    result_image_path ,p_list = detect_president(file_location)

    print("plist :",p_list[0])
    presiend_text = president_data[p_list[0]]

    print("text ",presiend_text)
    with open(result_image_path, "rb") as file_result:
        result_image =  file_result.read()

    if result_image :
        base64_image = base64.b64encode(result_image).decode('utf-8')
        print("64base image")

        result_data = {
            'message':presiend_text,
            'base64_image':base64_image,
            'image_path': result_image_path
        }

        return result_data
    else:
        
        return "errorr가 발생"
    #이미지 경로로 읽은 이미지와와 결과 컨텐츠를 보내줌





@fastapi_app.post("/process_ai_image_two/")
async def process_ai_image_two(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    print("!!!called ai_two====2222222222")

    if file1 and file2 :
        try:

            file_location1 = os.path.join(UPLOAD_SIMILARITY_IMAGE, file1.filename)
            file_location2 = os.path.join(UPLOAD_SIMILARITY_IMAGE, file2.filename)

            #print("file location 1", file_location1)
            #print("file location 2", file_location2)
            with open(file_location1, "wb") as file_object1:
                content = await file1.read()
                file_object1.write(content)

            with open(file_location2, "wb") as file_object2:
                content = await file2.read()
                file_object2.write(content)


            similarity_percent = compare_faces(file_location1, file_location2)


            result1 = f"두 사진의 유사도는 {similarity_percent:.2f}입니다\n"+get_similarity_ment(similarity_percent)
            return {'result':result1}

        except Exception as e:
            print("error server",e)
            result1 = "AI가 처리중에 에러가 발생하였습니다. 정확한 이미지를 올려주세요(1명 이미지)"
            return {'result':result1}

    #return  {'result':"error 발생"}

def get_similarity_ment(rate):

    import pandas as pd
    import random
    rate = int(rate)
    random_number = random.randint(1,49)
    data = pd.read_csv("similarity_text.csv", encoding='cp949')
    column_num = 0
    if rate < 30:
        column_num= 0
    elif rate>=30 and rate<60:
        column_num = 1
    elif rate>=60 and rate<80:
        column_num = 2
    else:
        column_num = 3

    result = data.iloc[random_number,column_num]
    print('result ment ', result)
    return  result


if __name__=="__main__":
    import uvicorn
    uvicorn.run(fastapi_app,host="0.0.0.0",port=8007)

