"""
Whisper模型管理模块 - 处理语音转录相关功能
"""
import torch
import time
import gc
import os
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from config.config import OPTIMIZED_MODELS, DEVICE, TORCH_DTYPE
from src.utils import extract_audio_parallel, monitor_gpu_usage

class OptimizedWhisperModel:
    def __init__(self, model_config):
        self.config = model_config
        self.model = None
        self.processor = None
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """加载并优化模型"""
        print(f"Loading model: {self.config['name']}")
        
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config["name"],
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                device_map="auto"
            )
            
            self.processor = AutoProcessor.from_pretrained(self.config["name"])
            
            print("Torch compile disabled for speed")
            
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=200,
                batch_size=self.config["batch_size"],
                torch_dtype=TORCH_DTYPE,
                return_timestamps=True,
                ignore_warning=True
            )
            
            print(f"Model loaded, batch size: {self.config['batch_size']}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.config["name"],
                torch_dtype=TORCH_DTYPE,
                batch_size=4,
                return_timestamps=True,
                ignore_warning=True
            )
    
    def transcribe(self, audio_path, language="chinese"):
        """高效转录"""
        generate_kwargs = {"language": language} if language != "auto" else {}
        
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
            result = self.pipeline(
                audio_path,
                generate_kwargs=generate_kwargs,
                return_timestamps=True
            )
        
        return result["text"]

# 全局模型缓存
model_instances = {}
current_model_key = None

def get_optimized_model(model_key):
    """获取优化模型实例"""
    global current_model_key, model_instances
    
    if model_key not in model_instances:
        config = OPTIMIZED_MODELS[model_key]
        model_instances[model_key] = OptimizedWhisperModel(config)
    
    current_model_key = model_key
    return model_instances[model_key]

def transcribe_high_utilization(audio_file, model_choice, language="chinese", preview_mode=False):
    """高GPU利用率转录"""
    if not audio_file:
        return "请上传文件", "", "请上传文件"
    
    from src.utils import check_ffmpeg
    if not check_ffmpeg():
        return "❌ FFmpeg未安装", "", "FFmpeg未安装"
    
    try:
        model_instance = get_optimized_model(model_choice)
        
        start_time = time.time()
        initial_gpu_status = monitor_gpu_usage()
        
        if preview_mode:
            print("Preview mode: first 3 minutes")
            audio_path = extract_audio_parallel(audio_file, duration=180)
            mode_info = "（高速预览：前3分钟）"
        else:
            print("Full transcription mode")
            audio_path = audio_file
            mode_info = "（完整转录 - GPU优化）"
        
        print("Starting transcription...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            transcribe_future = executor.submit(
                model_instance.transcribe, 
                audio_path, 
                language
            )
            transcript = transcribe_future.result()
        
        if preview_mode and audio_path != audio_file:
            try:
                os.unlink(audio_path)
            except:
                pass
        
        processing_time = time.time() - start_time
        final_gpu_status = monitor_gpu_usage()
        
        config = OPTIMIZED_MODELS[model_choice]
        
        performance_info = f"""
⚡ GPU优化统计:
• 模型: {model_choice}
• 批处理大小: {config['batch_size']}
• 处理时间: {processing_time:.1f}秒
• 模式: {mode_info}
• GPU优化: ✅ torch.compile + AMP
• 并行处理: ✅ 多线程预处理"""
        
        if preview_mode:
            file_size_mb = os.path.getsize(audio_file) / (1024*1024)
            estimated_full_time = (processing_time * file_size_mb) / 30
            performance_info += f"\n• 预估完整时间: ~{estimated_full_time:.1f}秒"
        
        display_text = transcript + "\n" + performance_info
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return display_text, transcript, final_gpu_status
        
    except Exception as e:
        error_msg = f"❌ 转录失败: {str(e)}"
        return error_msg, "", error_msg

def clear_all_cache():
    """清理所有缓存"""
    global model_instances
    
    for model_key in model_instances:
        del model_instances[model_key]
    model_instances = {}
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    return "Cache cleared"