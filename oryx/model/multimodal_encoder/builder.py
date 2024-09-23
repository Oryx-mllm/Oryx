import os
from .oryx_vit import OryxViTWrapper

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'vision_tower', getattr(vision_tower_cfg, 'mm_vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if "oryx_vit" in vision_tower:
        print(f"Buiding OryxViTWrapper from {vision_tower}...")
        path = vision_tower.split(":")[1]
        return OryxViTWrapper(vision_tower, path=path, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
