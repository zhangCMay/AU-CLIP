
cd /home/ubuntu/data/datasets/AU-CLIP-final_01/AU-CLIP


LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"


export CUDA_VISIBLE_DEVICES=0


nohup torchrun \
    --nproc_per_node=1 \
    --master_port=12355 \
    main_dfew.py \
    --config=configs/dfew/dfew.yaml \
    > ${LOG_FILE} 2>&1 &


echo "Training started with PID: $!"
echo "Log file: ${LOG_FILE}"
echo "Use 'tail -f ${LOG_FILE}' to monitor progress"
