import os
import requests
import json
import time
import re
from typing import Optional, Dict, Any, List
import yaml
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

class LocalLLMConnector:
    """本地LLM连接器类，支持vLLM部署的开源模型"""
    
    def __init__(self, config_path: str = "config/ilex_config.yaml"):
        self.config = self._load_config(config_path)
        
        # 从配置文件或环境变量获取设置
        self.model = os.getenv("LOCAL_MODEL", "Qwen3-8B_v2_p2")
        self.base_urls = self._get_base_urls()
        self.api_key = os.getenv("LOCAL_API_KEY", "EMPTY")
        self.timeout = int(os.getenv("LOCAL_TIMEOUT", "1000"))
        
        # 为每个URL创建OpenAI客户端
        self.clients = {}
        for url in self.base_urls:
            try:
                self.clients[url] = OpenAI(
                    api_key=self.api_key,
                    base_url=url,
                    timeout=self.timeout
                )
                print(f"✓ 成功连接到本地模型服务: {url}")
            except Exception as e:
                print(f"✗ 连接失败 {url}: {e}")
        
        if not self.clients:
            raise RuntimeError("无法连接到任何本地模型服务")
    
    def _get_base_urls(self) -> List[str]:
        """获取基础URL列表"""
        # 从环境变量获取URL列表
        urls_str = os.getenv("LOCAL_BASE_URLS")
        if urls_str:
            return [url.strip() for url in urls_str.split(",")]
        
        # 默认URL列表
        return [
            "http://localhost:8882/v1",
            "http://localhost:8883/v1"
        ]
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('ilex', {})
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def extract_content_and_json(self, input_text):
        """提取内容和JSON数据"""
        think_content, json_data = "", ""
        try:
            # 使用正则表达式提取思考内容和JSON数据
            think_match = re.search(r'<think>\n(.*?)\n</think>', input_text, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
                # 从原始文本中移除整个<think>...</think>块
                json_data = input_text.replace(think_match.group(0), '').strip().replace("```json\n","").replace("\n```","")
            else:
                # 如果没有找到<think>标签，则假定整个内容都是json数据
                think_content = ""
                json_data = input_text.strip().replace("```json\n","").replace("\n```","")
        except Exception as e:
            print(f"Error in extract_content_and_json: {str(e)}")

        return think_content, json_data

    def extract_json(self, output):
        """提取JSON数据"""
        try:
            data = json.loads(output)
            return data
        except json.JSONDecodeError:
            start = output.find('{')
            end = output.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            json_str = output[start:end]
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError:
                return None

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        调用本地LLM API
        
        Args:
            prompt: 输入提示词
            
        Returns:
            LLM生成的响应文本
        """
        system_prompt = "You are a helpful assistant."
        
        # 选择一个可用的客户端（轮询或随机选择）
        client = self._get_available_client()
        if not client:
            return "错误: 没有可用的模型服务"
        
        try:
            start_time = time.time()
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            
            duration = time.time() - start_time
            answer_dump = completion.model_dump()
            raw_answer = answer_dump.get("choices", [{}])[0].get("message",{}).get("content","")
            
            think_content, json_data = self.extract_content_and_json(raw_answer)
            
            # 如果成功提取到JSON数据，返回JSON，否则返回原始回答
            if json_data:
                return json_data
            else:
                return raw_answer
            
        except Exception as e:
            print(f"本地LLM API调用失败: {e}")
            return f"API调用失败: {str(e)}"
    
    def _get_available_client(self):
        """获取可用的客户端"""
        for url, client in self.clients.items():
            try:
                # 简单的连通性测试
                client.models.list()
                return client
            except Exception:
                continue
        return None
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            client = self._get_available_client()
            if not client:
                return False
            
            test_prompt = "请回复'连接正常'"
            response = self(test_prompt)
            return "连接正常" in str(response)
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False
    
    def batch_process(self, items: List[Dict], output_json: str, progress_file: str, 
                     concurrency: int = 10):
        """批量处理任务"""
        completed_tasks = self._load_progress(progress_file)
        
        remaining_tasks = [item for item in items if item["nid"] not in completed_tasks]
        
        # 轮询分配URL
        task_urls = [list(self.clients.keys())[i % len(self.clients)] for i in range(len(remaining_tasks))]
        
        all_nums = len(completed_tasks) + len(remaining_tasks)
        process_nums = len(remaining_tasks)
        succ_nums, fail_nums = 0, 0
        
        # 创建输出目录
        output_dir = os.path.dirname(output_json)
        os.makedirs(output_dir, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    self._process_single_item,
                    url,
                    item,
                    output_json,
                    progress_file
                ): (url, item)
                for url, item in zip(task_urls, remaining_tasks)
            }
            
            with tqdm(total=all_nums, desc="Processing", initial=len(completed_tasks), ncols=100) as pbar:
                for future in futures:
                    result = future.result()
                    
                    if result['succ']:
                        succ_nums += 1
                    else:
                        fail_nums += 1
                    
                    pbar.update(1)
        
        print(f"批量处理完成，总数: {process_nums}, 成功: {succ_nums}, 失败: {fail_nums}")
    
    def _process_single_item(self, url: str, item: Dict, output_json: str, progress_file: str):
        """处理单个项目"""
        client = self.clients[url]
        
        try:
            start_time = time.time()
            
            # 构建提示词
            prompt = f"QUERY: {item['item_title']}\nANSWER: {item['item_content'][:10000]}"
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            duration = time.time() - start_time
            answer_dump = completion.model_dump()
            raw_answer = answer_dump.get("choices", [{}])[0].get("message",{}).get("content","")
            
            think_content, json_data = self.extract_content_and_json(raw_answer)
            
            succ = bool(json_data)
            
            response = {
                "succ": succ,
                "loc": item["loc"],
                "nid": item["nid"],
                "item_content": item["item_content"],
                "item_title": item["item_title"],
                "answer": json_data if succ else raw_answer,
                "time_cost": duration,
            }
            
            # 保存进度和结果
            self._save_progress(progress_file, response)
            
            if succ:
                self._save_result(output_json, {
                    "loc": response["loc"],
                    "nid": response['nid'],
                    "answer": self.extract_json(response['answer'])
                })
            else:
                fail_json = output_json[:-5] + "_fail.json"
                self._save_result(fail_json, response)
            
            return response
            
        except Exception as e:
            print(f"处理项目失败 NID: {item.get('nid')}, URL: {url}, 错误: {e}")
            
            response = {
                "succ": False,
                "loc": item["loc"],
                "nid": item["nid"],
                "item_content": item["item_content"],
                "item_title": item["item_title"],
                "answer": f"Request Failed: {type(e).__name__}",
                "time_cost": time.time() - start_time,
            }
            
            self._save_progress(progress_file, response)
            return response
    
    def _load_progress(self, progress_file: str):
        """加载进度文件"""
        completed = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        nid = record.get("nid")
                        if isinstance(nid, list):
                            nid = nid[0]
                        if nid and record.get("succ") is True:
                            completed[nid] = record
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"跳过进度文件中格式错误的行: {line.strip()} - 错误: {e}")
        return completed
    
    def _save_progress(self, progress_file: str, result):
        """保存进度"""
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def _save_result(self, output_json: str, result):
        """保存结果"""
        with open(output_json, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def close(self):
        """关闭所有客户端连接"""
        for client in self.clients.values():
            try:
                client.close()
            except Exception:
                pass


# 测试函数
def test_local_llm_connector():
    """测试本地LLM连接器"""
    print("=== 测试本地LLM连接器 ===")
    
    try:
        print("正在初始化本地LLM连接器...")
        print("请确保vLLM服务已在 http://localhost:8882/v1 或 http://localhost:8883/v1 启动")
        connector = LocalLLMConnector()
        
        # 测试连接
        if connector.test_connection():
            print("✓ 本地LLM连接正常")
            
            # 测试SQL生成
            test_prompt = """
            请为以下问题生成SQL查询：
            问题：找出薪水最高的员工
            数据库：employees表包含id, name, salary, department_id字段
            请只返回SQL语句，不要包含其他解释。
            """
            
            response = connector(test_prompt)
            print(f"生成的响应: {response}")
        else:
            print("✗ 本地LLM连接失败")
            
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    test_local_llm_connector()