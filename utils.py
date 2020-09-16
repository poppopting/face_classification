import os
import time
import numpy as np
from tqdm.auto import tqdm
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import base64
import urllib
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class DataGenerator():
    def __init__(self, root_path, module, driver_path):
        self.root_path = root_path
        self.driver_path = driver_path
        self.module = module
        
    def get_idol_faces(self, keyword):
        
        image_path = os.path.join(self.root_path, keyword)
        if os.path.exists(image_path):
            pass
        else:
            os.mkdir(image_path)
          
        google_search_url = 'https://www.google.com/search?q={0}&site=webhp&tbm=isch'.format(keyword)
        
        chrome_options = Options() 
        chrome_options.add_argument('--headless')   #無頭模式
        driver = webdriver.Chrome(self.driver_path, options=chrome_options)
        print('Begin Searching...')
        driver.get(google_search_url)
        # 模擬滾輪
        for _ in range(7):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.close()
        print('Checking Search results...')
        cnt = 0
        for img in tqdm(soup.find_all('img')):
            get_src = img.get('src', 'None') 
            # 結果的格式分為 url 以及 base64
            if get_src.startswith('http'):
                image = self.url_to_image(get_src)

            elif get_src.startswith('data:image'):
                image = self.base64_to_image(get_src)
            # 只抓臉 其他不要
            catch_result = self.catch_face(image)
            if catch_result is not None:
                save_path = os.path.join(image_path, '{0}.jpg'.format(cnt))
                # 解決 cv2.imwrite 路徑不能用中文
                # 把圖片案jpg格式編碼 return (status, 編碼)
                cv2.imencode('.jpg', catch_result)[1].tofile(save_path)
                cnt += 1
     
        print('Download Completed, there are {0} available images.'.format(cnt))
        return self
    
    def base64_to_image(self, base):
        base64_file = base64.b64decode(base.split(',')[1])
        image_data = BytesIO(base64_file)
        image = np.asarray(Image.open(image_data), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def url_to_image(self, url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def catch_face(self, img):
        #列出資料集所有圖片
        
        #detectMultiScale(image, scaleFactor, minNeighbors, flags, minSize, maxSize)
        #scaleFactor 每次縮小比例
        #minNeighbors至 少要被檢測到幾次才算是真的目標
        #minSize, maxSize 目標最小最大尺寸
        faces = self.module.detectMultiScale(img, 1.3, 5)
        #只找抓到一個臉的
        if len(faces) == 1:
            x, y, w, h = faces[0]
            crop_img = img[y:y+h, x:x+w]
            
            return crop_img                            
        
        return None
    
def face_plot(model, picture, name_dic, transform, device):
    #設定字體
    font = cv2.FONT_HERSHEY_COMPLEX
    #顏色順序RGB
    im = Image.open(picture)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    #顏色順序BGR
    img = cv2.imread(picture)
    #return x,y 起點 w:x方向寬 h:y方向高
    faces = face_cascade.detectMultiScale(img)

    for x,y,w,h in faces:
        box = (x, y, x+w, y+h)
        #裁剪 只把臉截出來 retrun 圖
        crpim = im.crop(box).resize((64,64))
        #圖轉array
        target_image = transform(crpim).unsqueeze(0)
   
        res = model(target_image.to(device)).cpu().detach().numpy()
        #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)  線條寬度若為負 代表實心
        #cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        tag, color = name_dic[int(res.argmax(1))]
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, tag, (x + int(w/3)-70, y-10), font, 0.5, color, 1)
    plt.figure(figsize=(30,20))
    #BGR to RGB
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
