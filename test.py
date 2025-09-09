import requests
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_api_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 模型API配置 - 需要替换为实际的API信息
API_URL = "https://qianfan.baidubce.com/v2/chat/completions"
MODEL_NAME = "ernie-x1-turbo-32k" #"qwen2.5-7b-instruct"
API_KEY = "bce-v3/ALTAK-QoplnaBHAcEhU4Wy5qmjx/bbce25a048b589b1fef687f9fd49b8e491f4a882"  # 替换为实际的API密钥

class APIResponseError(Exception):
    """自定义API响应错误"""
    pass

from typing import Dict, Union, Any

def call_model_api(prompt: str, model_name: str = MODEL_NAME, temperature: float = 0.1, 
                  max_tokens: int = 500, max_retries: int = 3) -> Dict[str, Union[bool, str, float, Dict[str, Any]]]:
    """
    调用模型API的通用函数
    
    参数:
        prompt: 要发送给模型的提示文本
        model_name: 使用的模型名称
        temperature: 采样温度
        max_tokens: 生成的最大token数
        max_retries: 最大重试次数
        
    返回:
        包含API响应结果的字典，结构为:
        {
            "success": bool,
            "response": str,
            "parsed_result": dict,
            "time_cost": float,
            "request_id": str,
            "model": str,
            "error": str (仅当success为False时存在)
        }
    """
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            # 检查HTTP状态码
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    pass
                raise APIResponseError(error_msg)
            
            response_data = response.json()
            
            # 检查API响应是否包含有效内容
            if 'choices' not in response_data or not response_data['choices']:
                raise APIResponseError("API响应中缺少'choices'字段或内容为空")
            
            # 提取模型生成的内容
            answer_content = response_data["choices"][0]["message"]["content"]
            
            # 尝试解析JSON响应（如果返回的是JSON格式）
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
                "time_cost": round(time.time() - start_time, 2),
                "request_id": response_data.get("id", ""),
                "model": model_name
            }
            
        except APIResponseError as e:
            logging.warning(f"API响应错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"请求异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
        except Exception as e:
            logging.warning(f"未知错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
        
        # 如果不是最后一次尝试，则等待后重试
        if attempt < max_retries - 1:
            # 检查是否是限速错误
            if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                print(f"API限速，等待10秒后重试...")
                wait_time = 10  # 限速时固定等待10秒
            else:
                wait_time = 5 * (attempt + 1)  # 其他错误使用指数退避
            
            time.sleep(wait_time)
    
    return {
        "success": False,
        "error": f"达到最大重试次数({max_retries})后仍然失败",
        "prompt": prompt,
        "model": model_name
    }

def test_model_call():
    """测试模型调用的函数"""
    test_prompt = """
    请根据以下文本判断是否属于医美案例分享：
    
    <<content>>
    我在某医院做了双眼皮手术，术前是单眼皮，术后变成了明显的双眼皮，效果非常满意。医生根据我的脸型设计了手术方案，术后恢复期间按照医嘱护理，现在眼睛看起来更有神了。
    
    作为内容筛选助手，你需要判断上述文本内容是否属于医美案例分享内容。
    
    # 医美案例分享判定标准
    内容必须同时满足以下所有条件，任何一个不满足都判定为"否"：
    
    ## 核心要求（必须全部满足）
    - 必须明确包含具体的医美项目名称（如双眼皮、隆鼻、玻尿酸、热玛吉等）
    - 必须明确包含用户体验前后的效果对比说明
    - 可选包含用户诉求、真实体验描述、医生诊断方案、术后感受等
    
    ## 明确排除的内容
    - 医美知识科普或项目介绍（无个人体验）
    - 非医美项目分享（护肤品、按摩、健身等生美方式）
    - 非医美内容案例，如长痘、消肿等经验分享
    
    ## 评估要求
    1. 逐一检查每个核心要求
    2. 所有条件都满足才能判定为"是"
    3. 任何一个条件不满足立即判定为"否"
    
    请严格执行标准，输出json格式：
    {
        "name": "是否属于医美案例分享",
        "reason": "分别说明所有核心要求的满足情况，并给出最终判定理由",
        "result": "是" 或 "否"
    }
    """
    
    print("开始测试模型调用...")
    result = call_model_api(test_prompt)
    
    if result["success"]:
        print("\n测试成功！模型返回结果：")
        print("=" * 60)
        print(result["response"])
        print("=" * 60)
        print(f"\n解析后的结果: {result['parsed_result']}")
        print(f"\n请求耗时: {result['time_cost']}秒")
        print(f"请求ID: {result['request_id']}")
    else:
        print("\n测试失败！错误信息：")
        print(result["error"])
    
    return result

if __name__ == "__main__":
    # 在实际使用时，请确保:
    # 1. 替换API_KEY为有效的密钥
    # 2. 根据需要调整API_URL和MODEL_NAME
    # 3. 根据网络环境可能需要调整超时和重试参数
    
    test_result = test_model_call()
    
    # 并行测试示例（可选）
    print("\n开始并行测试（5个请求）...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(call_model_api, test_prompt) for _ in range(5)]
        
        for future in as_completed(futures):
            result: Dict[str, Union[bool, str, float, Dict[str, Any]]] = future.result()
            status = "成功" if result["success"] else "失败"
            print(f"并行请求完成 - 状态: {status}, 耗时: {result.get('time_cost', 'N/A')}秒")