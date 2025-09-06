import os
import requests
import json
from typing import Optional, Dict, Any
import yaml

class LLMConnector:
    """LLM连接器类"""
    
    def __init__(self, config_path: str = "config/ilex_config.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # 检查API密钥是否设置
        if not self.api_key:
            print("警告: OPENAI_API_KEY 环境变量未设置")
            print("请创建 .env 文件并设置你的 OpenAI API 密钥")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('ilex', {})
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 输入提示词
            
        Returns:
            LLM生成的响应文本
        """
        if not self.api_key:
            return "错误: 未设置API密钥。请在.env文件中设置OPENAI_API_KEY。"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的SQL查询生成助手。请根据用户的问题和数据库schema生成准确的SQL查询。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"LLM API调用失败: {e}")
            return f"API调用失败: {str(e)}"
        except Exception as e:
            print(f"处理响应时发生错误: {e}")
            return f"处理响应时发生错误: {str(e)}"
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            test_prompt = "请回复'连接正常'"
            response = self(test_prompt)
            return "连接正常" in response
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False


# 测试函数
def test_llm_connector():
    """测试LLM连接器"""
    print("=== 测试LLM连接器 ===")
    
    connector = LLMConnector()
    
    # 测试连接
    if connector.test_connection():
        print("✓ LLM API连接正常")
        
        # 测试SQL生成
        test_prompt = """
        请为以下问题生成SQL查询：
        问题：找出薪水最高的员工
        数据库：employees表包含id, name, salary, department_id字段
        """
        
        response = connector(test_prompt)
        print(f"生成的SQL: {response}")
    else:
        print("✗ LLM API连接失败")


if __name__ == "__main__":
    test_llm_connector()