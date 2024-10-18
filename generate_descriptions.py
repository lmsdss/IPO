import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
import torch.nn.functional as F

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
model.eval()

oxford_flowers_1_shot_image = ['data/oxford_flowers/jpg/image_01017.jpg',
                               'data/oxford_flowers/jpg/image_01381.jpg',
                               'data/oxford_flowers/jpg/image_01575.jpg',
                               'data/oxford_flowers/jpg/image_02248.jpg',
                               'data/oxford_flowers/jpg/image_02439.jpg',
                               'data/oxford_flowers/jpg/image_03137.jpg',
                               'data/oxford_flowers/jpg/image_03330.jpg',
                               'data/oxford_flowers/jpg/image_03404.jpg',
                               'data/oxford_flowers/jpg/image_03543.jpg',
                               'data/oxford_flowers/jpg/image_03743.jpg',
                               'data/oxford_flowers/jpg/image_03880.jpg',
                               'data/oxford_flowers/jpg/image_04021.jpg',
                               'data/oxford_flowers/jpg/image_04131.jpg',
                               'data/oxford_flowers/jpg/image_04248.jpg',
                               'data/oxford_flowers/jpg/image_04340.jpg',
                               'data/oxford_flowers/jpg/image_04587.jpg',
                               'data/oxford_flowers/jpg/image_04625.jpg',
                               'data/oxford_flowers/jpg/image_04937.jpg',
                               'data/oxford_flowers/jpg/image_04971.jpg',
                               'data/oxford_flowers/jpg/image_05093.jpg',
                               'data/oxford_flowers/jpg/image_05162.jpg',
                               'data/oxford_flowers/jpg/image_05215.jpg',
                               'data/oxford_flowers/jpg/image_05391.jpg',
                               'data/oxford_flowers/jpg/image_05611.jpg',
                               'data/oxford_flowers/jpg/image_05676.jpg',
                               'data/oxford_flowers/jpg/image_05730.jpg',
                               'data/oxford_flowers/jpg/image_05749.jpg',
                               'data/oxford_flowers/jpg/image_05825.jpg',
                               'data/oxford_flowers/jpg/image_06075.jpg',
                               'data/oxford_flowers/jpg/image_06173.jpg',
                               'data/oxford_flowers/jpg/image_06230.jpg',
                               'data/oxford_flowers/jpg/image_06324.jpg',
                               'data/oxford_flowers/jpg/image_06350.jpg',
                               'data/oxford_flowers/jpg/image_06429.jpg',
                               'data/oxford_flowers/jpg/image_06450.jpg',
                               'data/oxford_flowers/jpg/image_06490.jpg',
                               'data/oxford_flowers/jpg/image_06592.jpg',
                               'data/oxford_flowers/jpg/image_06650.jpg',
                               'data/oxford_flowers/jpg/image_06672.jpg',
                               'data/oxford_flowers/jpg/image_06746.jpg',
                               'data/oxford_flowers/jpg/image_06794.jpg',
                               'data/oxford_flowers/jpg/image_06834.jpg',
                               'data/oxford_flowers/jpg/image_06881.jpg',
                               'data/oxford_flowers/jpg/image_08073.jpg',
                               'data/oxford_flowers/jpg/image_06955.jpg',
                               'data/oxford_flowers/jpg/image_06981.jpg',
                               'data/oxford_flowers/jpg/image_07031.jpg',
                               'data/oxford_flowers/jpg/image_07117.jpg',
                               'data/oxford_flowers/jpg/image_08098.jpg',
                               'data/oxford_flowers/jpg/image_07171.jpg',
                               'data/oxford_flowers/jpg/image_07206.jpg']

with open("oxford_flowers_1_shot_image.txt", "w") as f:
    for i in range(501):
        image = Image.open(oxford_flowers_1_shot_image[i]).convert('RGB')
        question = 'Describe in detail the features that distinguish the scene in this image.'
        msgs = [{'role': 'user', 'content': question}]

        res, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )

        f.write("Image_" + str(i) + ":" + res + "\n")
