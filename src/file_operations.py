"""
文件操作模块 - 处理文件保存相关功能
"""
from src.utils import save_file_dialog
from config.config import DEFAULT_TRANSCRIPT_PREFIX, DEFAULT_SUMMARY_PREFIX

def save_transcript_with_dialog(text_with_info):
    """通过文件对话框保存转录结果"""
    # 提取纯文本
    if "⚡ GPU优化统计:" in text_with_info:
        pure_text = text_with_info.split("⚡ GPU优化统计:")[0].strip()
    else:
        pure_text = text_with_info
    
    return save_file_dialog(
        content=pure_text,
        title="保存转录文本",
        default_prefix=DEFAULT_TRANSCRIPT_PREFIX
    )

def save_summary_with_dialog(summary_text):
    """通过文件对话框保存AI总结结果"""
    return save_file_dialog(
        content=summary_text,
        title="保存AI总结",
        default_prefix=DEFAULT_SUMMARY_PREFIX
    )