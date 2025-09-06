#!/bin/bash

# vLLM部署脚本
# 用于启动本地模型服务

# 设置环境变量
export CUDA_VISIBLE_DEVICES="2,3"

# 模型路径配置
MODEL_PATH="/home/data02/xuxinyue/text_infer/model/Qwen3-8B_v2_p2/checkpoint-2000"

# 服务配置
PORT=8883
MODEL_NAME="Qwen3-8B_v2_p2"
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=16384

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请检查MODEL_PATH变量是否正确设置"
    exit 1
fi

# 检查vLLM是否已安装
if ! command -v vllm &> /dev/null; then
    echo "错误: vLLM未安装"
    echo "请先安装vLLM: pip install vllm"
    exit 1
fi

echo "正在启动vLLM服务..."
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "模型名称: $MODEL_NAME"
echo "张量并行大小: $TENSOR_PARALLEL_SIZE"
echo "最大模型长度: $MAX_MODEL_LEN"

# 启动vLLM服务
vllm serve "$MODEL_PATH" \
  --port "$PORT" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN"

echo "vLLM服务已启动"
echo "API端点: http://localhost:$PORT/v1"
echo "模型名称: $MODEL_NAME"