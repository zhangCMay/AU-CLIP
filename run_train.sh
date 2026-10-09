#!/bin/bash
cd /home/ubuntu/data/datasets/AU-CLIP-final_01/AU-CLIP

# 设置日志文件名（带时间戳）
LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=3

# 运行训练
nohup torchrun \
    --nproc_per_node=1 \
    --master_port=12355 \
    main1.py \
    --config=configs/dfew/dfew.yaml \
    > ${LOG_FILE} 2>&1 &

# 显示进程ID
echo "Training started with PID: $!"
echo "Log file: ${LOG_FILE}"
echo "Use 'tail -f ${LOG_FILE}' to monitor progress"
