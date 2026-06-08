import numpy as np
import cv2
import re
from PIL import Image
from google import genai

# ==========================================
# 模組 1：模擬 LLM 的分詞器 (Mock Tokenizer)
# 實務上這會是 HuggingFace 的 LlamaTokenizer
# ==========================================
class MockLLMTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size

# ==========================================
# 模組 2：OpenVLA 的核心 Action Tokenizer (連續轉離散)
# ==========================================
class ActionTokenizer:
    def __init__(self, tokenizer, bins: int = 256, min_action: float = 0.0, max_action: float = 1.0):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        
        # cut spaces  0.0 ~ 1.0 into 256 grids 
        self.bins_edges = np.linspace(min_action, max_action, self.n_bins +1 )
        # 算出每個區間的中心點
        # 0.0      0.25      0.5      0.75      1.0
        # |---------|---------|---------|---------|
        #     ↑          ↑         ↑         ↑
        # 0.125      0.375     0.625     0.875
        self.bin_centers = (self.bins_edges[:-1] + self.bins_edges[1:]) / 2.0

    def encode_to_token_ids(self, action: np.ndarray) -> np.ndarray:
        """【實體 -> 大腦】將連續動作轉換為 LLM 字典最後面的 Token ID"""
        action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        # 找出落在第幾個網格 (1 到 256)
        discretized_action = np.digitize(action, self.bins_edges)
        # 綁架字典最後的 Token (例如 32000 - 128 = 31872)
        token_ids = self.tokenizer.vocab_size - discretized_action
        return token_ids

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """【大腦 -> 實體】將 Token ID 還原為網格的中心點數值"""
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        # 確保索引不越界 (0 到 255)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        # 回傳該網格的正中心數值
        return self.bin_centers[discretized_actions]


    
# ==========================================
# 模組 3：結合 API 與 OpenVLA 邏輯的代理系統
# ==========================================
def run_openvla_simulation_agent(image_path, target_object, api_key):
    
    
    # 1. 取得圖片資訊
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 2. 使用 Gemini 取得「上帝視角」的連續座標 (0.000 ~ 1.000)
    print("\n[階段 1] 呼叫 VLM 視覺引擎，擷取環境空間特徵...")
    client = genai.Client(api_key=api_key)
    prompt = f"Find the bounding box of '{target_object}' in this image. Return ONLY a single list of four numbers in [ymin, xmin, ymax, xmax] format, scaled from 0.000 to 1.000. Do not add any extra text."
    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, image])
    raw_text = response.text.strip()
    
    # 提取浮點數 (例如 0.450)
    raw_coords = [float(x) for x in re.findall(r'0\.\d+', raw_text)]
    if len(raw_coords) != 4:
        print(f"視覺引擎輸出異常: {raw_text}")
        return

    continuous_action = np.array(raw_coords)
    print(f" -> 取得完美連續座標 (Ground Truth): {continuous_action}")
    # ------------------------------------------------------------------
    # 3. init OpenVLA Tokenizer
    mock_llm_vocab = MockLLMTokenizer(vocab_size=32000)

    action_tokenizer = ActionTokenizer(mock_llm_vocab, bins=256, min_action=0.0, max_action=1.0)

    # 4. 模擬 OpenVLA 的端到端預測流程
    print("\n[階段 2] 進入 OpenVLA Tokenization 流程...")
    
    # 【編碼】真實的 OpenVLA，模型吐出來是 Token ID
    predicted_token_ids = action_tokenizer.encode_to_token_ids(continuous_action)
    print(f" -> 模型大腦實際輸出的 Token IDs: {predicted_token_ids}")

    # 【解碼】系統將 Token ID 還原為機器手臂可以執行的網格中心點
    discrete_action = action_tokenizer.decode_token_ids_to_actions(predicted_token_ids)
    print(f" -> 還原為離散的物理執行座標 (Bin Centers): {discrete_action}")

    # 計算量化誤差 (Quantization Error)
    error = np.abs(continuous_action - discrete_action)
    print(f" -> 系統量化誤差 (空間精度流失): {error}")

    # 5. 視覺化執行結果
    ymin, xmin, ymax, xmax = discrete_action
    real_xmin, real_ymin = int(xmin * width), int(ymin * height)
    real_xmax, real_ymax = int(xmax * width), int(ymax * height)
    
    center_x = int((real_xmin + real_xmax) / 2)
    center_y = int((real_ymin + real_ymax) / 2)

    img_cv = cv2.imread(image_path)
    cv2.rectangle(img_cv, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
    cv2.drawMarker(img_cv, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    
    cv2.imwrite("openvla_sim_result.png", img_cv)
    print(f"\n[階段 3] 模擬完成！機器手臂抓取點已寫入 openvla_sim_result.png")




YOUR_API_KEY = "你的_API_KEY"
run_openvla_simulation_agent("my_desk.jpg", "mouse", YOUR_API_KEY)