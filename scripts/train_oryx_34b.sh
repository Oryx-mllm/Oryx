
export PYTHONPATH=/PATH/TO/oryx:$PYTHONPATH
VISION_TOWER='oryx_vit:PATH/TO/oryx_vit_new.pth'
DATA="PATH/TO/DATA.json"
FOLDER="PATH/TO/DATA"
MODEL_NAME_OR_PATH="PATH/TO/34B_MODEL"


export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x64"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=480
export VIDEO_MINRES=288

PROJECT_NAME=oryx_34b
echo ${PROJECT_NAME}
MASTER_ADDR='30.207.96.153'
nnode=1
nrank=$((INDEX%nnode))

torchrun  --nproc_per_node 8 --nnodes=$nnode --node_rank=$nrank --master_addr=$MASTER_ADDR --master_port=12343 \
    oryx/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --version qwen_1_5 \
    --data_path ${DATA} \
    --data_folder ${FOLDER} \
    --vision_tower $VISION_TOWER \
    --mm_projector_type simple_mlp_twoview \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bf16 True \
    --output_dir ./checkpoints/$PROJECT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --mm_resampler_type "dynamic_compressor" \
    --frames_upbound 64 \
    --report_to none