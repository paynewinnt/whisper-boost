"""
AI总结模块 - 处理DeepSeek AI文本总结功能
"""
import requests
import json
from config.config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL

def summarize_with_deepseek(text_with_info, user_api_key=""):
    """使用DeepSeek R1进行文本总结"""
    if not text_with_info or text_with_info.startswith("❌"):
        return "❌ 没有可总结的文本"
    
    # 优先使用代码中配置的API Key，如果为空则使用用户输入的
    api_key = DEEPSEEK_API_KEY if DEEPSEEK_API_KEY else user_api_key.strip()
    
    if not api_key:
        return "❌ 请输入DeepSeek API Key或在代码中配置"
    
    # 提取纯文本
    if "⚡ GPU优化统计:" in text_with_info:
        pure_text = text_with_info.split("⚡ GPU优化统计:")[0].strip()
    else:
        pure_text = text_with_info
    
    if not pure_text or len(pure_text.strip()) < 50:
        return "❌ 文本内容太短，无法进行有效总结"
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 构建提示词
        prompt = f"""请对以下转录文本进行简洁的总结，要求：
1. 提取主要观点和关键信息
2. 保持逻辑清晰，条理分明
3. 总结长度控制在原文的1/3以内
4. 使用中文输出

转录文本：
{pure_text}

请开始总结："""
        
        data = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "stream": False
        }
        
        # 发送请求
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                summary = result["choices"][0]["message"]["content"].strip()
                
                # 添加总结信息
                summary_info = f"""
📝 AI总结 (DeepSeek R1):
{summary}

---
📊 总结统计:
• 原文长度: {len(pure_text)} 字符
• 总结长度: {len(summary)} 字符
• 压缩比例: {len(summary)/len(pure_text)*100:.1f}%
• 模型: DeepSeek R1 Reasoner
"""
                return summary_info
            else:
                return "❌ API返回格式错误"
        else:
            error_detail = ""
            try:
                error_info = response.json()
                if "error" in error_info:
                    error_detail = f": {error_info['error'].get('message', '')}"
            except:
                pass
            return f"❌ API请求失败 (状态码: {response.status_code}){error_detail}"
            
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请检查网络连接"
    except requests.exceptions.RequestException as e:
        return f"❌ 网络请求失败: {str(e)}"
    except Exception as e:
        return f"❌ 总结失败: {str(e)}"