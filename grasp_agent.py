import torch
from PIL import Image
import cv2
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
import os

from huggingface_hub import login
# from dotenv import load_dotenv

# load_dotenv()
# login(os.getenv("MY_HF_TOKEN"))


def run_grasp_agent(image_path, target_object):
    # ==========================================
    # 1. 輕量化VLM brain (以開源的 PaliGemma 
    # ==========================================

    
    model_id = "google/paligemma-3b-pt-448" # 也可以換成其他支援邊界框的輕量模型
    
    print("正在初始化大腦模型...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # ==========================================
    # 2. 看與聽：準備圖片與「特製的提示詞 (Prompt)」
    # ==========================================
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 這是最關鍵的步驟！用嚴格的 Prompt 逼大腦只能吐出座標，不要講廢話
    prompt = f"detect {target_object}" 
    #(個人學習)註：有些模型適合用 "Where is the {target_object}? Output in [ymin, xmin, ymax, xmax] format."
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 3. 大腦思考與寫出密碼：執行模型推論
    # ==========================================
    print(f"大腦正在尋找目標物：{target_object}...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    # 將大腦輸出的 Token 密碼還原成文字解讀
    result_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"大腦回傳的原始字串: {result_text}")

    # ==========================================
    # 4. 手臂執行（模擬）：解析字串並用 OpenCV 畫出抓取點
    # ==========================================
    # 假設模型吐出標準的標準化座標格式，例如 "[312, 450, 600, 700]" (ymin, xmin, ymax, xmax)
    # 我們用正則表達式把數字抓出來
    coordinates = [int(x) for x in re.findall(r'\d+', result_text)]
    
    if len(coordinates) >= 4:
        ymin, xmin, ymax, xmax = coordinates
        
        # 將標準化座標 (0~1000) 還原成圖片的真實像素位置
        real_xmin = int(xmin * width / 1000)
        real_ymin = int(ymin * height / 1000)
        real_xmax = int(xmax * width / 1000)
        real_ymax = int(ymax * height / 1000)
        
        # 計算物件的「中心點數值」，作為實體手臂要抓取的目標
        center_x = int((real_xmin + real_xmax) / 2)
        center_y = int((real_ymin + real_ymax) / 2)
        
        # 使用 OpenCV 讀取圖片並進行視覺化標記
        img_cv = cv2.imread(image_path)
        # 畫出綠色的物件邊界框
        cv2.rectangle(img_cv, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
        # 畫出紅色的十字準星（代表手臂抓取點中心）
        cv2.drawMarker(img_cv, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        # 儲存結果圖
        output_path = "grasp_result.png"
        cv2.imwrite(output_path, img_cv)
        print(f"成功！結果已儲存至 {output_path}，中心抓取點座標為: [{center_x}, {center_y}]")
    else:
        print("抱歉，大腦這次沒有吐出正確的座標格式，請調整 Prompt 再試一次。")

# 測試執行（你可以自己拿手機拍一張桌上有滑鼠或飲料的圖片來測試）



run_grasp_agent("my_desk.jpg", "mouse")