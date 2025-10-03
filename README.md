## Task 1

# Vision Transformer (ViT) on CIFAR-10

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

| Model      | Test Accuracy |
| ---------- | ------------- |
| ViT (best) | **90.10%**    |


