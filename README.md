# AI-Algorithms-to-Detect-Explosives-in-Waste-Receptables-Using-Vision-Transformers
# IED Detection in Waste Receptacles Using Vision Transformers

## Project Overview
This project addresses the critical problem of detecting improvised explosive devices (IEDs) hidden in public waste receptacles to enhance public safety. Traditional detection methods have limitations, and this work explores the use of **Vision Transformer (ViT)** models for image-based classification of IEDs versus non-IED waste.

Using an augmented TrashNet dataset combined with additional IED images, we formulated a **binary classification task** to distinguish between IED-containing images and various types of waste.

## What We Have Done
- **Dataset Preparation:**
  - Utilized an augmented TrashNet dataset containing 3041 images: 514 IED images and 2527 non-IED images (cardboard, glass, metal, paper, plastic, general trash).
  - Resized all images to 224x224 pixels for compatibility with the ViT model.
  - Split the dataset into 70% training, 15% validation, and 15% testing subsets.

- **Model Architecture:**
  - Implemented the Vision Transformer (ViT) b-32 model, leveraging pre-trained weights for transfer learning.
  - Adapted ViT, originally developed for NLP, to process images by dividing them into patches with positional encoding, followed by transformer encoder layers.
  - Used a softmax output layer and cross-entropy loss function for binary classification.

- **Training Setup:**
  - Trained the model for 10 epochs on an NVIDIA Tesla T4 GPU using Google Colab.
  - Optimized with the Adam optimizer and a learning rate of 0.001.
  - Performed hyperparameter tuning using the validation set to prevent overfitting.

- **Evaluation Metrics:**
  - Precision: 44%
  - Recall: 42%
  - F1 Score: 43%
  - Accuracy: 93%
  
  *Note:* The high accuracy reflects the model’s ability to classify non-IED images correctly due to class imbalance, while precision and recall indicate room for improvement in detecting IEDs specifically.

- **Results:**
  - Achieved a validation accuracy of up to 94.95% and a testing accuracy of 93.03% after 10 epochs.
  - Observed fluctuations in validation loss indicating potential overfitting, addressed through hyperparameter tuning.
  - Demonstrated that ViT can effectively generalize for this task, though further improvements are needed to enhance detection reliability.

## Experimental Environment
- Hardware: NVIDIA Tesla T4 GPU, 16 GB RAM, Intel Xeon CPU
- Platform: Google Colab
- Libraries: PyTorch/TensorFlow (assumed), ViT pre-trained weights

## Future Work
- Expand and balance the dataset with more varied IED and non-IED samples.
- Explore advanced data augmentation and ensemble learning methods.
- Experiment with other model architectures or fine-tune hyperparameters further.
- Improve precision and recall to reduce false positives and negatives in real-world scenarios.

## Acknowledgments
Thanks to Dr. Amoakoh Gyasi-Agyei for continuous support throughout this project.

---

This project demonstrates the potential of Vision Transformers in enhancing public safety by detecting explosives in waste management systems.

