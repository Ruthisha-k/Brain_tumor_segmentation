# Brain Tumor Segmentation using Computer Vision 🧠🖥️

This project is a deep learning-based brain tumor classification system using Convolutional Neural Networks (CNNs). It classifies MRI images into four categories: **glioma**, **meningioma**, **pituitary**, and **no tumor**. A simple Gradio interface is also provided for real-time prediction.

---

## 🔍 Project Overview

Brain tumor detection is crucial for early diagnosis and treatment. This project implements a computer vision model trained on MRI scans to automate the classification process.

---

## 📁 Dataset

- Located in Google Drive
- Organized into 4 folders:
  - `glioma/`
  - `meningioma/`
  - `notumor/`
  - `pituitary/`
- Images are resized to **128x128** before training.

---

## 🧠 Model Architecture

A CNN model built using TensorFlow/Keras with the following layers:

- `Conv2D + ReLU` → `MaxPooling2D` → `BatchNormalization`
- Repeated with increasing filters (32 → 64 → 128)
- `GlobalAveragePooling2D`
- `Dense (128)` with `ReLU` + `Dropout`
- Output `Dense` layer with `Softmax`

---

## 📊 Training Configuration

- **Image Size:** 128x128  
- **Batch Size:** 16  
- **Epochs:** 10  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Data Split:** 80% training, 20% testing

---

## 🚀 How to Run

1. **Mount Google Drive** (dataset stored there)
2. **Install required libraries**
   ```bash
   pip install gradio
   ```
3. **Train the model** (runs on Google Colab)
4. **Launch Gradio app**
   ```python
   interface.launch()
   ```

---

## 🖼️ Gradio Interface

- Upload an MRI image
- Get prediction: **glioma**, **meningioma**, **pituitary**, or **notumor**
- Built with Gradio for easy user interaction

---

## 🧪 Sample Prediction Function

```python
def predict_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    label = lb.inverse_transform(prediction)
    return label[0]
```

---

## 📌 Requirements

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- scikit-learn
- Gradio
- Google Colab (recommended)

---

## 📈 Future Improvements

- Use transfer learning (e.g., EfficientNet, ResNet)
- Add segmentation support
- Add Grad-CAM for interpretability
- Improve UI and deploy with Flask or Streamlit

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- Dataset: [Kaggle / Public MRI Repositories]
- Frameworks: TensorFlow, Gradio, scikit-learn
