import os

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "openbmb/MiniCPM-V-2"
OUTPUT_PATH = "desc/oxford_flowers_1_shot_image.txt"

oxford_flowers_1_shot_image = [
    "data/oxford_flowers/jpg/image_01017.jpg",
    "data/oxford_flowers/jpg/image_01381.jpg",
    "data/oxford_flowers/jpg/image_01575.jpg",
    "data/oxford_flowers/jpg/image_02248.jpg",
    "data/oxford_flowers/jpg/image_02439.jpg",
    "data/oxford_flowers/jpg/image_03137.jpg",
    "data/oxford_flowers/jpg/image_03330.jpg",
    "data/oxford_flowers/jpg/image_03404.jpg",
    "data/oxford_flowers/jpg/image_03543.jpg",
    "data/oxford_flowers/jpg/image_03743.jpg",
    "data/oxford_flowers/jpg/image_03880.jpg",
    "data/oxford_flowers/jpg/image_04021.jpg",
    "data/oxford_flowers/jpg/image_04131.jpg",
    "data/oxford_flowers/jpg/image_04248.jpg",
    "data/oxford_flowers/jpg/image_04340.jpg",
    "data/oxford_flowers/jpg/image_04587.jpg",
    "data/oxford_flowers/jpg/image_04625.jpg",
    "data/oxford_flowers/jpg/image_04937.jpg",
    "data/oxford_flowers/jpg/image_04971.jpg",
    "data/oxford_flowers/jpg/image_05093.jpg",
    "data/oxford_flowers/jpg/image_05162.jpg",
    "data/oxford_flowers/jpg/image_05215.jpg",
    "data/oxford_flowers/jpg/image_05391.jpg",
    "data/oxford_flowers/jpg/image_05611.jpg",
    "data/oxford_flowers/jpg/image_05676.jpg",
    "data/oxford_flowers/jpg/image_05730.jpg",
    "data/oxford_flowers/jpg/image_05749.jpg",
    "data/oxford_flowers/jpg/image_05825.jpg",
    "data/oxford_flowers/jpg/image_06075.jpg",
    "data/oxford_flowers/jpg/image_06173.jpg",
    "data/oxford_flowers/jpg/image_06230.jpg",
    "data/oxford_flowers/jpg/image_06324.jpg",
    "data/oxford_flowers/jpg/image_06350.jpg",
    "data/oxford_flowers/jpg/image_06429.jpg",
    "data/oxford_flowers/jpg/image_06450.jpg",
    "data/oxford_flowers/jpg/image_06490.jpg",
    "data/oxford_flowers/jpg/image_06592.jpg",
    "data/oxford_flowers/jpg/image_06650.jpg",
    "data/oxford_flowers/jpg/image_06672.jpg",
    "data/oxford_flowers/jpg/image_06746.jpg",
    "data/oxford_flowers/jpg/image_06794.jpg",
    "data/oxford_flowers/jpg/image_06834.jpg",
    "data/oxford_flowers/jpg/image_06881.jpg",
    "data/oxford_flowers/jpg/image_08073.jpg",
    "data/oxford_flowers/jpg/image_06955.jpg",
    "data/oxford_flowers/jpg/image_06981.jpg",
    "data/oxford_flowers/jpg/image_07031.jpg",
    "data/oxford_flowers/jpg/image_07117.jpg",
    "data/oxford_flowers/jpg/image_08098.jpg",
    "data/oxford_flowers/jpg/image_07171.jpg",
    "data/oxford_flowers/jpg/image_07206.jpg",
]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because this script loads MiniCPM-V-2 on GPU.")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model.eval()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    question = "Describe in detail the features that distinguish the scene in this image."

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, image_path in enumerate(oxford_flowers_1_shot_image):
            if not os.path.exists(image_path):
                print(f"[{i + 1}/{len(oxford_flowers_1_shot_image)}] Skip missing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            msgs = [{"role": "user", "content": question}]

            with torch.no_grad():
                res, context, _ = model.chat(
                    image=image,
                    msgs=msgs,
                    context=None,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.7,
                )

            f.write(f"Image_{i}:{res}\n")
            f.flush()

            print(f"[{i + 1}/{len(oxford_flowers_1_shot_image)}] Wrote description for {image_path}")

    print(f"Done. Descriptions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
