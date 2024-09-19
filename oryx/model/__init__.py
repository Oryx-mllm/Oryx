import os

AVAILABLE_MODELS = {
    "oryx_llama": "OryxLlamaForCausalLM, OryxConfig",
    "oryx_qwen": "OryxQwenForCausalLM, OryxQwenConfig",
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        raise e
        print(f"Failed to import {model_name} from llava.language_model.{model_name}")
