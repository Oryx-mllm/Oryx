export HF_HOME=""

export GPT_EVAL_VERSION=""
export OPENAI_API_KEY=""
export OPENAI_API_URL=""

export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x64"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=480
export VIDEO_MINRES=288

MODEL_NAME=""
MODEL_PATH=""

accelerate launch --num_processes=1 -m lmms_eval --model oryx --model_args pretrained=$MODEL_PATH,max_frames_num=64,mm_resampler_type="dynamic_compressor"  --tasks videochatgpt_temporal  --batch_size 1 --log_samples --log_samples_suffix eval --output_path ./logs_eval/$MODEL_NAME --verbosity DEBUG