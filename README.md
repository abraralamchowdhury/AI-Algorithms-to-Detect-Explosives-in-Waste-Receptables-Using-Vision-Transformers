# 💣🗑️ IED Detection in Waste Receptacles Using Vision Transformers 🚀

## 📋 Project Overview
This project tackles the critical challenge of detecting **improvised explosive devices (IEDs)** hidden in public waste bins to enhance public safety. Traditional detection methods have limitations, so we explored **Vision Transformer (ViT)** models for image-based IED detection.

We created a **binary classification task** using an augmented TrashNet dataset and additional IED images to distinguish between IED-containing and regular waste images.

## ✅ What We Have Done

- 🗂️ **Dataset Preparation:**
  - Used an augmented TrashNet dataset with **3041 images**: 514 IED and 2527 non-IED (cardboard, glass, metal, paper, plastic, general trash).
  - Resized all images to **224x224 pixels** for ViT compatibility.
  - Split  **70% training**, **15% validation**, **15% testing**.

- 🧠 **Model Architecture:**
  - Implemented **ViT b-32**, leveraging pre-trained weights for transfer learning.
  - Adapted ViT to process images as patches with positional encoding and transformer encoder layers.
  - Used a **softmax output layer** and **cross-entropy loss** for binary classification.

- 🏋️ **Training Setup:**
  - Trained for **10 epochs** on an **NVIDIA Tesla T4 GPU** (Google Colab).
  - Used **Adam optimizer** (learning rate: 0.001).
  - Tuned hyperparameters using the validation set to prevent overfitting.

- 📊 **Evaluation Metrics:**
  - **Precision:** 44%
  - **Recall:** 42%
  - **F1 Score:** 43%
  - **Accuracy:** 93%
  
  > ⚠️ *High accuracy reflects strong non-IED classification due to class imbalance; precision/recall indicate room for improved IED detection.*

- 🏆 **Results:**
  - Validation accuracy up to **94.95%**, test accuracy **93.03%** after 10 epochs.
  - Monitored validation loss for overfitting; tuned hyperparameters accordingly.
  - Showed ViT’s potential for this safety-critical task, with further improvements needed.

## 🖥️ Experimental Environment
- **Hardware:** NVIDIA Tesla T4 GPU, 16 GB RAM, Intel Xeon CPU
- **Platform:** Google Colab
- **Libraries:** PyTorch/TensorFlow (ViT pre-trained weights)

## 🔭 Future Work
- Expand and balance the dataset with more diverse IED/non-IED samples.
- Explore advanced data augmentation and ensemble methods.
- Experiment with other architectures and hyperparameter fine-tuning.
- Focus on boosting precision and recall to minimize real-world false positives/negatives.

## 🙏 Acknowledgments
Special thanks to **Dr. Amoakoh Gyasi-Agyei** for ongoing support throughout this project.

---

This project demonstrates the promise of **Vision Transformers** in boosting public safety by detecting explosives in waste management systems. 🛡️🧠
