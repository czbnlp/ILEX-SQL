# ILEX-SQL: 双模式智能SQL生成系统

ILEX-SQL是一个集成LPE-SQL经验复用能力和Amber迭代探索能力的双模式智能SQL生成系统。

## 项目结构

- `config/` - 系统配置文件
  - `ilex_config.yaml` - ILEX-SQL系统配置
- `prompts/ilex/` - 系统提示词模板
  - `complexity_analysis.txt` - 复杂度分析提示词
  - `experience_extraction.txt` - 经验提取提示词
  - `problem_decompose.txt` - 问题分解提示词
  - `result_synthesize.txt` - 结果合成提示词
  - `subquestion_solve.txt` - 子问题解决提示词
- `src/ilex_core/` - 核心模块
  - `__init__.py` - 模块初始化
  - `execution_memory.py` - 执行记忆管理
  - `experience_upgrader.py` - 经验升级器
  - `exploration_engine.py` - 探索引擎
  - `mode_selector.py` - 模式选择器
  - `problem_decomposer.py` - 问题分解器
- `src/knowledge_base/` - 知识库
  - `init_correct_set.json` - 初始正确案例集
  - `init_mistake_set.json` - 初始错误案例集

## 功能特性

- **双模式处理**：根据问题复杂度智能选择经验模式或探索模式
- **经验复用**：利用历史成功案例快速生成SQL查询
- **迭代探索**：对复杂问题进行分步推理和探索
- **知识升级**：从成功探索中提取可复用经验模式
- **记忆管理**：维护执行过程中的中间结果和状态
