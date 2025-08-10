"""
配置文件 - 存储所有配置常量和设置
"""
import torch

# DeepSeek API配置
DEEPSEEK_API_KEY = "xxxxx"  # 如果为空，用户可在页面上输入
# DEEPSEEK_API_KEY = ""  # 如果为空，用户可在页面上输入

# 设备配置
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 高利用率模型配置 - 优化速度
OPTIMIZED_MODELS = {
    "Small (GPU优化)": {
        "name": "openai/whisper-small",
        "batch_size": 16
    },
    "Medium (高利用率)": {
        "name": "openai/whisper-medium", 
        "batch_size": 8
    },
    "Large-v3 (最大化GPU)": {
        "name": "openai/whisper-large-v3",
        "batch_size": 4
    }
}

# API配置
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"

# 文件配置
DEFAULT_TRANSCRIPT_PREFIX = "transcript_optimized"
DEFAULT_SUMMARY_PREFIX = "ai_summary"
SUPPORTED_FILE_TYPES = ["audio", "video"]

# 界面配置
APP_TITLE = "GPU高利用率Whisper"
APP_PORT = 7862
MAX_FILE_SIZE = "1GB"

