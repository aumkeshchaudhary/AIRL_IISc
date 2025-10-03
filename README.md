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
