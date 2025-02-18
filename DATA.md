# Data

### Data Format

We follow the data format below, which is similar to LLaVA. You can directly use the original file path or pack the multi-modal files into patches following [create_patch.py](). Patch is a binary file containing continuous image or video files in byte format, which may accelerate reading speed in some cases.


- Image Data:

```
[
    {
        'id': ID of the data
        'image': ***.png (path to the image file or positions in patches)
        'conversations': [{"from": "human", "value": "<image>\n"}, {"from": "gpt", "value": ""}]
    }
]
```

The format for image patch is:

```
{
    "patch": "patch_00000",
    "start_num": 846113989,
    "size": 27141
}
```

- Video Frame Data:

```
[
    {
        'id': ID of the data
        'video': ***.mp4 (path to the video file or positions in patches)
        'conversations': [{"from": "human", "value": "<image>\n"}, {"from": "gpt", "value": ""}]
    }
]
```

The format for video patch is:

```
{
    "patch": "patch_000000",
    "size": [ 5605, 8902, 7917, 5562, 9249, 8785, 8379, 10389, 10505, 10337, 8481, 8164, 5562, 8844, 10565, 8035, 7768, 8969, 5643, 10478, 7632, 10980, 9986, 3602, 2848, 7591, 10766, 7813, 5605, 9840, 9664, 5605, 7726, 4828, 8006, 5562, 9711, 7903, 9542, 10626, 8827, 11268, 11115, 1832, 11354, 9222, 3965, 10426, 10427, 7311, 9726, 7655, 10025, 5350, 10098, 10470, 4877, 10273, 9730, 10150, 5604, 7203, 9881, 2246, 11114, 3790, 5567, 10490, 4072, 1701],
    "start_num": 26608266
}
```

### Instruction for Oryx Data

#### **Oryx-SFT-Data**

1. Download [Oryx-SFT-Data](https://huggingface.co/datasets/THUdyh/Oryx-SFT-Data) from huggingface. 

2. Download all the [video patches](https://huggingface.co/datasets/THUdyh/Oryx-SFT-Data/tree/main) from huggingface. 

3. Check and modify the video patch path in the json to the true path in your machine.
