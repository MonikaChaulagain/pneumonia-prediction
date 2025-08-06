### ✅ `README.md` — Pneumonia Detection Using ResNet18

# 🫁 Pneumonia Detection with ResNet18 (PyTorch)

This project implements a Convolutional Neural Network (CNN) using transfer learning with ResNet18 to classify chest X-ray images into two classes: **Normal** and **Pneumonia**.

## 📁 Dataset

- Source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Structure:


chestXray/
├── chest\_xray/
│   ├── train/
│   ├── test/
│   └── val/   # (optional - not used in current version)


- The model uses a stratified 80/20 split from the `train/` directory for training and validation.
- Final evaluation is done on the separate `test/` directory.



## 🧠 Model Architecture

- **Backbone**: Pretrained ResNet18 (ImageNet)
- **Final Layer**: Modified to output 2 classes with dropout for regularization
- **Loss Function**: Weighted CrossEntropyLoss to handle class imbalance
- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Stops training if validation loss does not improve for 5 epochs

## 🔧 Data Preprocessing

### ✅ Training Transform:
- Random resized crop
- Affine transformations (rotation, translation, scale)
- Random horizontal flip
- Convert grayscale to 3-channel
- Normalization (ImageNet mean/std)

### ✅ Validation & Test Transform:
- Resize to 224x224
- Grayscale to 3-channel
- Normalization (ImageNet mean/std)

## 📊 TensorBoard Logging

Training/validation loss and accuracy are logged for every epoch. 

## 🏁 Training & Evaluation

* Training runs for up to **30 epochs**
* Early stopping is triggered based on validation loss
* Best model (`best_resnet_pneumonia.pth`) is saved automatically


## 📉 Results

| Metric            | Score (%) |
| ----------------- | --------- |
| Train Accuracy    | \~99      |
| Val Accuracy      | \~97      |
| **Test Accuracy** | \~93      |



## 🚀 Future Improvements

* Apply Grad-CAM for visual explainability
* Implement confidence thresholding
* Try lighter models for deployment (e.g., MobileNetV2, EfficientNet)
* Experiment with semi-supervised learning or pseudo-labeling

## 📜 License

This project is open for educational and non-commercial use.

## 🙋‍♀️ Author

Built by a CSIT student learning deep learning through real-world projects. Contributions and feedback welcome!

