from locust import HttpUser, between, task,SequentialTaskSet

class UserBehavior(SequentialTaskSet):

    @task(1)
    def login(self):

        response = self.client.get('/common/login/')
        csrf_token = response.cookies.get('csrftoken')  # CSRF 토큰 추출

        #print("response login :", response.status_code)
        payload = {
            'username': 'test4',
            'password': 'ocode1234',
            'csrfmiddlewaretoken' : csrf_token  # CSRF 토큰 추가
        }
        login_response = self.client.post('/common/login/',data=payload)

        if login_response.status_code ==200:
            print("login successful")
            self.client.cookies.update(login_response.cookies)
        else:
            print("login fail   :",login_response.status_code)

    @task(1)
    def create_post(self):

        response_csrf = self.client.get('/pybo/detection/post/create/')
        csrf_token = response_csrf.cookies.get('csrftoken')  # CSRF 토큰 추출
        print("response_csrf url :",response_csrf.url)

        payload = {
            'subject': '여기는 부하테스트중입니다',
            'content': "여기는 부하중",

        }

        headers = {
            'X-CSRFToken': csrf_token  # CSRF 토큰을 헤더에 추가

        }

        with open("3.png", 'rb') as f:
            files = {'image1':f}
            response = self.client.post('/pybo/detection/post/create/',data=payload,files=files,headers=headers)
            print("response url :", response.url)
            print("status code create post: ", response.status_code)
            #print("Response content:", response.content.decode('utf-8'))
            #print("Response headers:", response.headers)

           #http://3.34.71.98/pybo/detection/post/create/



class UserBehavior2(SequentialTaskSet):

    @task(1)
    def login_s(self):

        response = self.client.get('/common/login/')
        csrf_token = response.cookies.get('csrftoken')  # CSRF 토큰 추출

        #print("response login :", response.status_code)
        payload = {
            'username': 'test4',
            'password': 'ocode1234',
            'csrfmiddlewaretoken' : csrf_token  # CSRF 토큰 추가
        }
        login_response = self.client.post('/common/login/',data=payload)

        if login_response.status_code ==200:
            print("login successful similarity")
            self.client.cookies.update(login_response.cookies)
        else:
            print("login fail   :",login_response.status_code)



    @task(1)
    def create_post_similarity(self):

        response_csrf = self.client.get('/pybo/similarity/post/create/')
        csrf_token = response_csrf.cookies.get('csrftoken')  # CSRF 토큰 추출
        #print("response_csrf url :",response_csrf.url)

        payload = {
            'subject': '여기는 유사도 도배중입니다.',
            'content': "여기는 도배중",

        }
        headers = {
            'X-CSRFToken': csrf_token  # CSRF 토큰을 헤더에 추가
        }

        with open("1_s.png", 'rb') as f1 ,open("2_s.png",'rb') as f2:
            files = {'image1':f1,'image2':f2}
            response = self.client.post('/pybo/similarity/post/create/',data=payload,files=files,headers=headers)
            print("response url :", response.url)
            print("status code create post similarity: ", response.status_code)
            #print("Response content:", response.content.decode('utf-8'))
            #print("Response headers:", response.headers)

           #http://3.34.71.98/pybo/detection/post/create/




class UserTest(HttpUser):
    tasks = [UserBehavior,UserBehavior2]
    wait_time = between(1, 3)