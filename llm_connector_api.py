#!/usr/bin/env python3
"""
API模型连接器类
基于test.py中的API调用逻辑，提供与LocalLLMConnector相同的接口
支持百度千帆等API模型的调用
"""

import os
import requests
import json
import time
import logging
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
# yaml 库已不再需要，可以移除

class APIResponseError(Exception):
    """自定义API响应错误"""
    pass

class APILLMConnector:
    """API模型连接器类，支持百度千帆等API模型"""
    
    def __init__(self):
        """
        初始化API连接器 (使用硬编码配置)
        """
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        
        # ==================== 硬编码配置 START ====================
        # 直接在这里设置您的API信息
        self.api_url = "https://qianfan.baidubce.com/v2/chat/completions"
        self.model_name = "ernie-4.5-turbo-128k"
        self.api_key = "bce-v3/ALTAK-QoplnaBHAcEhU4Wy5qmjx/bbce25a048b589b1fef687f9fd49b8e491f4a882"
        
        # 其他参数也使用固定的默认值
        self.timeout = 120
        self.max_retries = 3
        self.temperature = 0.1
        self.max_tokens = 500
        # ===================== 硬编码配置 END =====================
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0.0
        }
        self.stats_lock = Lock()
        
        # 验证API密钥
        if not self.api_key:
            self.logger.error("API密钥未设置！请在代码中直接填写 self.api_key 的值。")
    
    # _load_config 方法已不再需要，已删除

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        调用模型API的主要接口
        
        Args:
            prompt: 输入提示文本
            **kwargs: 其他参数，如temperature, max_tokens等
            
        Returns:
            模型生成的文本内容
        """
        result = self.call_model_api(prompt, **kwargs)
        
        if result["success"]:
            return result["response"]
        else:
            self.logger.error(f"API调用失败: {result.get('error', '未知错误')}")
            return f"API调用失败: {result.get('error', '未知错误')}"
    
    def call_model_api(self, prompt: str, model_name: str = None, temperature: float = None, 
                      max_tokens: int = None, max_retries: int = None) -> Dict[str, Any]:
        """
        调用模型API的通用函数
        """
        model_name = model_name or self.model_name
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        max_retries = max_retries or self.max_retries
        
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        with self.stats_lock:
            self.stats['total_requests'] += 1
        
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # ==================== 代码修改处 START ====================
                # 定义一个空的代理字典，以忽略系统/终端中设置的任何HTTP/HTTPS代理
                no_proxies = {
                    "http": None,
                    "https": None,
                }

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    proxies=no_proxies  # <-- 添加此参数以禁用代理
                )
                # ===================== 代码修改处 END =====================
                
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.reason}"
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail}"
                    except:
                        error_msg += f" - {response.text}"
                    raise APIResponseError(error_msg)
                
                response_data = response.json()
                
                if 'choices' not in response_data or not response_data['choices']:
                    raise APIResponseError("API响应中缺少'choices'字段或内容为空")
                
                answer_content = response_data["choices"][0]["message"]["content"]
                time_cost = time.time() - start_time
                
                with self.stats_lock:
                    self.stats['successful_requests'] += 1
                    self.stats['total_time'] += time_cost
                
                try:
                    clean_content = answer_content.strip()
                    if clean_content.startswith("```json"):
                        clean_content = clean_content[7:]
                    if clean_content.endswith("```"):
                        clean_content = clean_content[:-3]
                    clean_content = clean_content.strip()
                    parsed_result = json.loads(clean_content)
                except (json.JSONDecodeError, AttributeError):
                    parsed_result = {"raw_response": answer_content}
                
                return {
                    "success": True,
                    "response": answer_content,
                    "parsed_result": parsed_result,
                    "time_cost": round(time_cost, 2),
                    "request_id": response_data.get("id", ""),
                    "model": model_name
                }
                
            except (APIResponseError, requests.exceptions.RequestException) as e:
                last_error = e
                self.logger.warning(f"API调用错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            except Exception as e:
                last_error = e
                self.logger.warning(f"未知错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                error_str = str(last_error)
                if "rate_limit_exceeded" in error_str.lower() or "429" in error_str:
                    self.logger.info(f"API限速，等待10秒后重试...")
                    wait_time = 10
                else:
                    wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
        
        with self.stats_lock:
            self.stats['failed_requests'] += 1
        
        return {
            "success": False,
            "error": f"达到最大重试次数({max_retries})后仍然失败. Last error: {str(last_error)}",
            "prompt": prompt,
            "model": model_name
        }
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
            result = self.call_model_api(test_prompt, max_tokens=10, max_retries=1)
            
            if result["success"]:
                self.logger.info(f"✓ API模型连接测试成功: {self.model_name}")
                return True
            else:
                self.logger.warning(f"✗ API模型连接测试失败: {result.get('error', '未知错误')}")
                return False
        except Exception as e:
            self.logger.error(f"✗ API模型连接测试异常: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['average_time'] = stats['total_time'] / stats['successful_requests'] if stats['successful_requests'] > 0 else 0
        else:
            stats['success_rate'] = 0.0
            stats['average_time'] = 0.0
        return stats
    
    def batch_call(self, prompts: List[str], max_workers: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """批量调用API（支持并发）"""
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.call_model_api, prompt, **kwargs): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"批量调用时，任务 {index} 失败: {e}")
                    results[index] = {
                        "success": False,
                        "error": str(e),
                        "prompt": prompts[index]
                    }
        return results

# 测试函数
def test_api_connector():
    """测试API连接器"""
    print("=== 测试API模型连接器 (硬编码配置) ===")
    
    # 创建连接器 (不再需要传入任何参数)
    connector = APILLMConnector()
    
    # 测试连接
    print("\n1. 测试API连接...")
    if connector.test_connection():
        print("✓ API连接测试通过")
    else:
        print("✗ API连接测试失败，请检查硬编码的API Key和网络连接。")
        return
    
    # 测试单个调用
    print("\n2. 测试单个API调用...")
    test_prompt = "请生成一个查询所有员工信息的SQL语句，只返回SQL语句。"
    result = connector(test_prompt)
    print(f"生成的SQL: {result}")
    
    # 测试批量调用
    print("\n3. 测试批量API调用...")
    test_prompts = [
        "生成查询员工总数的SQL语句",
        "生成查询最高薪水的SQL语句", 
        "生成查询IT部门员工的SQL语句"
    ]
    
    batch_results = connector.batch_call(test_prompts, max_workers=2)
    for i, res in enumerate(batch_results):
        if res["success"]:
            print(f"提示{i+1}: ✓ 成功，响应: {res['response'].strip()}")
        else:
            print(f"提示{i+1}: ✗ 失败，错误: {res.get('error', '未知错误')}")
    
    # 打印统计信息
    print("\n4. 统计信息:")
    stats = connector.get_stats()
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功请求数: {stats['successful_requests']}")
    print(f"失败请求数: {stats['failed_requests']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"平均响应时间: {stats['average_time']:.2f}秒")
    
    print("\n=== API连接器测试完成 ===")

if __name__ == "__main__":
    # 设置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 运行测试
    test_api_connector()