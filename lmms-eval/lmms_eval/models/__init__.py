import os
import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

AVAILABLE_MODELS = {
    "oryx": "Oryx",
    "oryx_image": "OryxImage",
    "from_log": "FromLog",
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError as e:
        print(model_name, e)
