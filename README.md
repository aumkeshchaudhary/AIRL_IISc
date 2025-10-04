# Task 1

## Vision Transformer (ViT) on CIFAR-10

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/ViT_Model)

This repository implements a Vision Transformer (ViT) trained on the CIFAR-10 dataset.

---

##  How to Run in Colab

1. Open the notebook in Google Colab.
2. Ensure GPU is enabled:  
   `Runtime` → `Change runtime type` → select **GPU**.
3. Install dependencies:
   ```bash
   !pip install -q torch torchvision tqdm einops
4. Run all cells in sequence.
   Training will take ~2–3 hours for 300 epochs on a T4 GPU.

---
## Best Model Config
        image_size: 32
        patch_size: 4
        in_channels: 3
        num_classes: 10
        emb_dim: 192        # model width
        num_heads: 6
        depth: 8            # transformer layers
        mlp_ratio: 4.0
        drop: 0.1
        batch_size: 128
        epochs: 300
        lr: 3e-4
        weight_decay: 0.05
        warmup_epochs: 5
        seed: 42
        augmentations: RandomCrop + HorizontalFlip + AutoAugment (CIFAR10 policy) + Normalize

---
## Results

| Epoch | Train Loss | Train Accuracy (%) | Val Loss | Val Accuracy (%) |
|-------|------------|---------------------|----------|------------------|
| 50   | 0.7818     | 72.45              | 0.5411   | 80.93            |
| 100  | 0.5178     | 81.62              | 0.4174   | 86.32            |
| 150  | 0.3777     | 86.74              | 0.4312   | 87.19            |
| 200  | 0.2875     | 89.99              | 0.3988   | 89.16            |
| 250  | 0.2338     | 91.88              | 0.4187   | 89.41            |
| 300  | 0.2211     | 92.09              | 0.4017   | 89.96            |


--- 
## Bonus Analysis

* Patch size: Smaller patches (4×4) preserved local detail, helping on CIFAR-10 (small images).

* Depth vs. Width: Depth=8, width=192 balanced expressiveness and efficiency — deeper models risk overfitting.

* Augmentation: AutoAugment + RandomCrop significantly boosted generalization vs. plain normalization.

* Optimizer & schedule: AdamW with warmup and weight decay stabilized training, improving convergence.

---

## Notes

* The model achieves strong performance (~90%) while being relatively lightweight.

* Further improvements could come from larger embeddings, Mixup/CutMix, or longer training schedules.



# Task 2

## Text-Driven Image & Video Segmentation with SAM 2 + GroundingDINO

This project integrates GroundingDINO for text-prompted object detection and Segment Anything 2 (SAM 2) for high-quality segmentation on both images and videos.

## Features

* Image Segmentation: Provide a text prompt (e.g., "cat") to detect and segment objects.

* Video Segmentation: Track and segment objects throughout a video using SAM 2 video predictor.

* Colab-Friendly: Full pipeline runs in Google Colab.

## Installation (Colab)

        !pip install git+https://github.com/facebookresearch/segment-anything-2.git -q
        !pip install supervision transformers groundingdino-py -q
        !pip install opencv-python matplotlib pillow numpy torch torchvision -q

        # Download checkpoints
        !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
        !wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
        !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

## Example Usage

   * Image Segmentation

          image_url = "https://wpcdn.web.wsu.edu/news/uploads/sites/2797/2025/03/cat2.jpg"
          text_prompt = "cat"
          result = text_driven_segmentation(image_url, text_prompt)

  * Video Segmentation

          from google.colab import files
          uploaded = files.upload()
          video_path = list(uploaded.keys())[0]

          text_prompt = "dog"
          output_video = text_driven_video_segmentation(video_path, text_prompt)

## Demo Video
![output_video (2)](https://github.com/user-attachments/assets/ca41646e-249f-4359-b484-f0eaf0d2fea4)

       



