"""
工具函数模块 - 系统检测和通用工具函数
"""
import subprocess
import torch
import os
import tempfile
import time
from tkinter import filedialog
import tkinter as tk

def check_ffmpeg():
    """检查FFmpeg是否可用"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def get_gpu_info():
    """获取GPU信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, gpu_name, f"{gpu_memory:.1f}GB"
    return False, "None", "0GB"

def monitor_gpu_usage():
    """监控GPU使用率"""
    if not torch.cuda.is_available():
        return "CPU模式"
    
    try:
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_percent = (memory_used / memory_total) * 100
        
        return f"GPU内存: {memory_used:.1f}GB/{memory_total:.1f}GB ({memory_percent:.1f}%)"
    except:
        return "GPU状态获取失败"

def extract_audio_parallel(video_path, start_time=0, duration=None):
    """并行音频提取"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-threads', '0',
                '-preset', 'ultrafast',
                '-y'
            ]
            
            if duration:
                cmd.extend(['-t', str(duration)])
                
            cmd.append(temp_audio.name)
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                return temp_audio.name
            else:
                print(f"FFmpeg warning: {stderr.decode()}")
                return video_path
                
    except Exception as e:
        print(f"音频提取失败: {e}")
        return video_path

def save_file_dialog(content, title, default_prefix):
    """通用文件保存对话框"""
    if not content or content.startswith("❌"):
        return "❌ 没有可保存的内容"
    
    if "正在生成" in content:
        return "❌ 内容尚未完成，请等待"
    
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        default_filename = f"{default_prefix}_{int(time.time())}.txt"
        
        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )
        
        root.destroy()
        
        if not file_path:
            return "❌ 用户取消保存"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return f"✅ 文件已保存到: {file_path}"
        
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"