#!/usr/bin/env python3
"""
统一的LLM连接器
支持本地vLLM部署和API调用两种方式
通过配置自动选择合适的连接器
"""

import os
import logging
from typing import Optional, Dict, Any, List
from llm_connector_local import LocalLLMConnector
from llm_connector_api import APILLMConnector

class UnifiedLLMConnector:
    """统一的LLM连接器，支持多种模型访问方式"""
    
    def __init__(self, 
                 config_path: str = "config/ilex_config.yaml",
                 connector_type: str = None,
                 **kwargs):
        """
        初始化统一的LLM连接器
        
        Args:
            config_path: 配置文件路径
            connector_type: 连接器类型 ('local', 'api', 'auto')
            **kwargs: 其他参数，传递给具体的连接器
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 确定连接器类型
        self.connector_type = connector_type or self._detect_connector_type()
        
        # 初始化具体的连接器
        self.connector = self._create_connector(config_path, **kwargs)
        
        self.logger.info(f"初始化统一LLM连接器，类型: {self.connector_type}")
    
    def _detect_connector_type(self) -> str:
        """
        自动检测连接器类型
        
        Returns:
            连接器类型 ('local', 'api')
        """
        # 检查API相关环境变量
        return "api"
        # api_url = os.getenv("API_URL")
        # api_key = os.getenv("API_KEY")
        
        # if api_url and api_key:
        #     return "api"
        
        # 检查本地vLLM相关环境变量
        local_urls = os.getenv("LOCAL_BASE_URLS")
        
        if local_urls:
            return "local"
        
        # 默认使用本地连接器
        return "local"
    
    def _create_connector(self, config_path: str, **kwargs):
        """
        创建具体的连接器实例
        
        Args:
            config_path: 配置文件路径
            **kwargs: 其他参数
            
        Returns:
            连接器实例
        """
        if self.connector_type == "api":
            self.logger.info("使用API连接器")
            return APILLMConnector()
        elif self.connector_type == "local":
            self.logger.info("使用本地vLLM连接器")
            return LocalLLMConnector(config_path, **kwargs)
        else:
            raise ValueError(f"不支持的连接器类型: {self.connector_type}")
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        调用模型的主要接口
        
        Args:
            prompt: 输入提示文本
            **kwargs: 其他参数
            
        Returns:
            模型生成的文本内容
        """
        return self.connector(prompt, **kwargs)
    
    def test_connection(self) -> bool:
        """
        测试连接（统一接口）
        
        Returns:
            是否连接成功
        """
        try:
            return self.connector.test_connection()
        except AttributeError:
            # 如果具体连接器没有test_connection方法，尝试基本调用
            try:
                test_result = self.connector("Hello, this is a test.")
                return len(test_result) > 0
            except Exception as e:
                self.logger.error(f"连接测试失败: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息（统一接口）
        
        Returns:
            统计信息字典
        """
        try:
            return self.connector.get_stats()
        except AttributeError:
            # 如果具体连接器没有get_stats方法，返回基础统计
            return {
                "connector_type": self.connector_type,
                "status": "active"
            }
    
    def batch_call(self, prompts: List[str], max_workers: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """
        批量调用（统一接口）
        
        Args:
            prompts: 提示文本列表
            max_workers: 最大并发数
            **kwargs: 其他参数
            
        Returns:
            结果列表
        """
        try:
            return self.connector.batch_call(prompts, max_workers, **kwargs)
        except AttributeError:
            # 如果具体连接器没有batch_call方法，使用顺序调用
            self.logger.warning("当前连接器不支持批量调用，使用顺序调用")
            results = []
            for prompt in prompts:
                try:
                    response = self.connector(prompt, **kwargs)
                    results.append({
                        "success": True,
                        "response": response,
                        "prompt": prompt
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "prompt": prompt
                    })
            return results
    
    def switch_connector(self, new_type: str, **kwargs):
        """
        切换连接器类型
        
        Args:
            new_type: 新的连接器类型 ('local', 'api')
            **kwargs: 新连接器的参数
        """
        if new_type not in ['local', 'api']:
            raise ValueError(f"不支持的连接器类型: {new_type}")
        
        old_type = self.connector_type
        self.connector_type = new_type
        
        try:
            self.connector = self._create_connector(**kwargs)
            self.logger.info(f"连接器已从 {old_type} 切换到 {self.connector_type}")
        except Exception as e:
            self.logger.error(f"切换连接器失败: {e}")
            # 恢复原连接器
            self.connector_type = old_type
            raise
    
    def get_connector_info(self) -> Dict[str, Any]:
        """
        获取连接器信息
        
        Returns:
            连接器信息字典
        """
        info = {
            "connector_type": self.connector_type,
            "connector_class": self.connector.__class__.__name__
        }
        
        # 添加具体连接器的信息
        if hasattr(self.connector, 'model_name'):
            info['model'] = self.connector.model_name
        elif hasattr(self.connector, 'model'):
            info['model'] = self.connector.model
        
        if hasattr(self.connector, 'api_url'):
            info['api_url'] = self.connector.api_url
        
        if hasattr(self.connector, 'base_urls'):
            info['base_urls'] = self.connector.base_urls
        
        return info


# 测试函数
def test_unified_connector():
    """测试统一的LLM连接器"""
    print("=== 测试统一的LLM连接器 ===")
    
    # 测试自动检测
    print("\n1. 测试自动检测连接器类型...")
    connector = UnifiedLLMConnector()
    print(f"检测到的连接器类型: {connector.connector_type}")
    print(f"连接器信息: {connector.get_connector_info()}")
    
    # 测试连接
    print("\n2. 测试连接...")
    if connector.test_connection():
        print("✓ 连接测试成功")
    else:
        print("✗ 连接测试失败")
        return
    
    # 测试单个调用
    print("\n3. 测试单个调用...")
    test_prompt = "请生成一个查询所有员工信息的SQL语句。"
    result = connector(test_prompt)
    print(f"生成的SQL: {result}")
    
    # 测试统计信息
    print("\n4. 测试统计信息...")
    stats = connector.get_stats()
    print(f"统计信息: {stats}")
    
    # 测试批量调用（如果支持）
    print("\n5. 测试批量调用...")
    test_prompts = [
        "生成查询员工总数的SQL",
        "生成查询最高薪水的SQL",
        "生成查询IT部门员工的SQL"
    ]
    
    batch_results = connector.batch_call(test_prompts, max_workers=2)
    print(f"批量调用完成，结果数: {len(batch_results)}")
    
    print("\n=== 统一连接器测试完成 ===")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_unified_connector()