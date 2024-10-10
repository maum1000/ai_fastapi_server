
from locust import HttpUser, between,task

class FastAPIUser(HttpUser):
    wait_time = between(1,3)


    @task
    def upload_files1(self):
        with open("3.png",'rb') as f:
            response = self.client.post("/process_ai_image/", files={"file":f})
            print("response statecode ", response.status_code)


    @task
    def upload_files2(self):
        with open("1_s.png", 'rb') as f1, open("2_s.png",'rb') as f2:
            response = self.client.post("/process_ai_image_two/", files={"file1": f1, "file2":f2})
            print("response statecode  process_ai_image_two", response.status_code)

