"""
执行记忆管理模块
管理探索模式中的中间结果和执行状态
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ExecutionRecord:
    """执行记录数据类"""
    subquestion: str
    result: Any
    sql: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = 0.0
    step_number: int = 0
    success: bool = True
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()
        self.success = self.error is None

class ExecutionMemory:
    """执行记忆管理类"""
    
    def __init__(self, max_size: int = 10, config_path: str = "../config/ilex_config.yaml"):
        """
        初始化执行记忆
        
        Args:
            max_size: 记忆最大容量
            config_path: 配置文件路径
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.max_size = max_size
        self.memory: List[ExecutionRecord] = []
        self.current_step = 0
        
        # 加载配置
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.config = config.get('ilex', {}).get('memory', {})
        except Exception as e:
            self.config = {}
            self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        
        # 配置参数
        self.context_window_size = self.config.get('context_window_size', 5)
        self.retain_successful_paths = self.config.get('retain_successful_paths', True)
        self.retain_failed_paths = self.config.get('retain_failed_paths', False)
    
    def add(self, 
            subquestion: str, 
            result: Any, 
            sql: Optional[str] = None, 
            error: Optional[str] = None,
            execution_time: float = 0.0) -> None:
        """
        添加执行记录
        
        Args:
            subquestion: 子问题
            result: 执行结果
            sql: SQL查询（可选）
            error: 错误信息（可选）
            execution_time: 执行时间
        """
        record = ExecutionRecord(
            subquestion=subquestion,
            result=result,
            sql=sql,
            error=error,
            execution_time=execution_time,
            step_number=self.current_step,
            timestamp=time.time()
        )
        
        self.memory.append(record)
        self.current_step += 1
        
        # 维护记忆大小
        self._maintain_memory_size()
        
        self.logger.info(f"添加执行记录: 步骤 {record.step_number}, 子问题: {subquestion[:50]}...")
        if error:
            self.logger.warning(f"执行失败: {error}")
    
    def _maintain_memory_size(self) -> None:
        """维护记忆大小"""
        if len(self.memory) > self.max_size:
            # 根据配置决定保留策略
            if self.retain_successful_paths and not self.retain_failed_paths:
                # 只保留成功的记录
                self.memory = [r for r in self.memory if r.success][-self.max_size:]
            elif not self.retain_successful_paths and self.retain_failed_paths:
                # 只保留失败的记录
                self.memory = [r for r in self.memory if not r.success][-self.max_size:]
            else:
                # 默认策略：保留最新的记录
                self.memory = self.memory[-self.max_size:]
    
    def get_recent_records(self, count: int = None) -> List[ExecutionRecord]:
        """
        获取最近的执行记录
        
        Args:
            count: 获取记录数量，默认为上下文窗口大小
            
        Returns:
            最近的执行记录列表
        """
        if count is None:
            count = self.context_window_size
        
        return self.memory[-count:] if count > 0 else []
    
    def get_context_for_prompt(self, max_length: int = 2000) -> str:
        """
        生成用于提示的上下文
        
        Args:
            max_length: 最大长度限制
            
        Returns:
            格式化的上下文字符串
        """
        recent_records = self.get_recent_records()
        
        context_parts = []
        current_length = 0
        
        for record in reversed(recent_records):  # 从最新的记录开始
            record_text = self._format_record_for_prompt(record)
            
            if current_length + len(record_text) > max_length:
                break
                
            context_parts.append(record_text)
            current_length += len(record_text)
        
        # 反转回正确的时间顺序
        context_parts.reverse()
        
        return "\n\n".join(context_parts)
    
    def _format_record_for_prompt(self, record: ExecutionRecord) -> str:
        """
        格式化记录用于提示
        
        Args:
            record: 执行记录
            
        Returns:
            格式化的记录字符串
        """
        parts = [
            f"步骤 {record.step_number}:",
            f"子问题: {record.subquestion}",
            f"结果: {record.result}"
        ]
        
        if record.sql:
            parts.append(f"SQL: {record.sql}")
        
        if record.error:
            parts.append(f"错误: {record.error}")
        
        return " | ".join(parts)
    
    def get_successful_results(self) -> List[Tuple[str, Any]]:
        """
        获取所有成功的执行结果
        
        Returns:
            成功结果的列表，每个元素为(子问题, 结果)元组
        """
        return [(r.subquestion, r.result) for r in self.memory if r.success]
    
    def get_failed_records(self) -> List[ExecutionRecord]:
        """
        获取所有失败的执行记录
        
        Returns:
            失败记录列表
        """
        return [r for r in self.memory if not r.success]
    
    def find_result_by_subquestion(self, subquestion: str) -> Optional[Any]:
        """
        根据子问题查找结果
        
        Args:
            subquestion: 子问题
            
        Returns:
            找到的结果，如果不存在则返回None
        """
        for record in reversed(self.memory):  # 从最新的记录开始查找
            if record.subquestion == subquestion and record.success:
                return record.result
        return None
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            统计信息字典
        """
        if not self.memory:
            return {"total_steps": 0}
        
        total_steps = len(self.memory)
        successful_steps = len([r for r in self.memory if r.success])
        failed_steps = total_steps - successful_steps
        
        total_execution_time = sum(r.execution_time for r in self.memory)
        average_execution_time = total_execution_time / total_steps if total_steps > 0 else 0
        
        return {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time,
            "current_step": self.current_step
        }
    
    def clear_memory(self) -> None:
        """清空记忆"""
        self.memory.clear()
        self.current_step = 0
        self.logger.info("执行记忆已清空")
    
    def save_to_file(self, filepath: str) -> None:
        """
        保存记忆到文件
        
        Args:
            filepath: 文件路径
        """
        data = {
            "memory": [asdict(record) for record in self.memory],
            "current_step": self.current_step,
            "timestamp": time.time()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"执行记忆已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存执行记忆失败: {e}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        从文件加载记忆
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.memory = [ExecutionRecord(**record) for record in data["memory"]]
            self.current_step = data.get("current_step", 0)
            
            self.logger.info(f"执行记忆已从文件加载: {filepath}")
        except Exception as e:
            self.logger.error(f"加载执行记忆失败: {e}")
    
    def is_question_resolved(self, original_question: str) -> bool:
        """
        判断原始问题是否已解决
        
        Args:
            original_question: 原始问题
            
        Returns:
            是否已解决
        """
        # 简单的实现：检查是否有足够的执行步骤
        # 在实际应用中，这里需要更复杂的逻辑
        if not self.memory:
            return False
        
        # 检查最近的步骤是否成功
        recent_records = self.get_recent_records(3)
        if not recent_records:
            return False
        
        # 如果最近的几个步骤都成功了，认为问题可能已解决
        recent_success_rate = sum(1 for r in recent_records if r.success) / len(recent_records)
        return recent_success_rate >= 0.8
    
    def get_summary(self) -> str:
        """
        获取执行记忆摘要
        
        Returns:
            摘要字符串
        """
        if not self.memory:
            return "执行记忆为空"
        
        stats = self.get_execution_statistics()
        summary_parts = [
            f"总步骤数: {stats['total_steps']}",
            f"成功步骤: {stats['successful_steps']}",
            f"失败步骤: {stats['failed_steps']}",
            f"成功率: {stats['success_rate']:.2%}",
            f"平均执行时间: {stats['average_execution_time']:.2f}秒"
        ]
        
        return " | ".join(summary_parts)
    
    def __len__(self) -> int:
        """返回记忆中的记录数量"""
        return len(self.memory)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"ExecutionMemory(steps={len(self.memory)}, current_step={self.current_step})"


# 测试函数
def test_execution_memory():
    """测试执行记忆功能"""
    print("=== 执行记忆测试 ===")
    
    # 创建执行记忆
    memory = ExecutionMemory(max_size=5)
    
    # 添加测试记录
    test_records = [
        ("找出薪水最高的经理", "经理ID: 123", "SELECT manager_id FROM managers ORDER BY salary DESC LIMIT 1"),
        ("找出经理123所在的部门", "部门ID: D001", "SELECT department_id FROM managers WHERE manager_id = 123"),
        ("找出部门D001的所有员工", "员工数: 15", "SELECT COUNT(*) FROM employees WHERE department_id = 'D001'")
    ]
    
    for subquestion, result, sql in test_records:
        memory.add(subquestion, result, sql, execution_time=0.5)
    
    # 测试各种功能
    print(f"记忆摘要: {memory.get_summary()}")
    print(f"上下文: {memory.get_context_for_prompt()}")
    print(f"统计信息: {memory.get_execution_statistics()}")
    print(f"是否已解决: {memory.is_question_resolved('找出与薪水最高的经理在同一个部门的所有员工')}")
    
    # 测试文件保存和加载
    memory.save_to_file("test_memory.json")
    new_memory = ExecutionMemory()
    new_memory.load_from_file("test_memory.json")
    print(f"加载后的记忆: {new_memory.get_summary()}")


if __name__ == "__main__":
    test_execution_memory()