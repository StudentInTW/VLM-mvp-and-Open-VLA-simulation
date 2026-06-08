from google import genai
from google.genai import types
from PIL import Image
import cv2
import re

def run_api_grasp_agent(image_path, target_object, api_key):
  
    print("正在連接新版雲端大腦...")
    client = genai.Client(api_key=api_key)

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"找不到圖片 {image_path}，請確認檔名是否正確。")
        return
        
    width, height = image.size

    # 要求模型以 [ymin, xmin, ymax, xmax] 格式輸出標準化座標 (0~1000)
    prompt = f"Find the bounding box of '{target_object}' in this image. Return ONLY a single list of four numbers in [ymin, xmin, ymax, xmax] format, scaled from 0 to 1000. Do not add any markdown, explanation, or extra text."

    print(f"大腦正在尋找目標物：{target_object}...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt, image]
        )

        
        result_text = response.text.strip()
        print(f"大腦回傳的原始字串: {result_text}")
    except Exception as e:
        print(f"推論過程發生錯誤: {e}")
        return


    coordinates = [int(x) for x in re.findall(r'\d+', result_text)]
    
    if len(coordinates) == 4:
        ymin, xmin, ymax, xmax = coordinates

        real_xmin = int(xmin * width / 1000)
        real_ymin = int(ymin * height / 1000)
        real_xmax = int(xmax * width / 1000)
        real_ymax = int(ymax * height / 1000)
        
        center_x = int((real_xmin + real_xmax) / 2)
        center_y = int((real_ymin + real_ymax) / 2)
        
        #  OpenCV read 並進行視覺化標記
        img_cv = cv2.imread(image_path)
        cv2.rectangle(img_cv, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
        cv2.drawMarker(img_cv, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        output_path = "grasp_result_api.png"
        cv2.imwrite(output_path, img_cv)
        print(f"成功！結果已儲存至 {output_path}，中心抓取點座標為: [{center_x}, {center_y}]")
    else:
        print("大腦沒有吐出標準的 4 個座標數字，請重新執行或調整 Prompt。")



from dotenv import load_dotenv
import os
load_dotenv()

YOUR_API_KEY = os.getenv("GEMENI_API_KEY") 
run_api_grasp_agent("my_desk.jpg", "mouse", YOUR_API_KEY)