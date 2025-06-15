# 🦴 Osteoporosis Knee Classification App
![man-knee-pain-he-puts-260nw-2476578973 jpg](https://github.com/user-attachments/assets/69d8db39-531e-417d-a6a5-79578d115f0a)

# 📌 Overview
This project is a dual-mode diagnostic tool that classifies knees as either healthy or affected by osteoporosis, using:

• Structured patient data

• Knee X-ray images

The application combines traditional machine learning and deep learning techniques, wrapped in a user-friendly Streamlit interface. It was developed to support orthopedic professionals in accelerating diagnosis and making more informed decisions.


# 💡 Business Context
An orthopedic doctor approached me to develop a predictive tool to assist in early detection of osteoporosis in knees—using both patient medical data and X-ray scans. This model was designed to be fast, interpretable, and practical for deployment in a clinical setting.

# ⚙️ How It Works
Users can choose between two classification modes:

1. Patient Data Classification
Inputs: Demographic, lifestyle, and medical information
Model: Logistic Regression
Output: Binary classification — Healthy or Osteoporotic

2. Image-Based Classification
Input: Knee X-ray image (JPEG/PNG)
Model: Convolutional Neural Network (CNN)
Output: Binary classification — Healthy or Osteoporotic

📍 Try the app here:
👉 [Live Demo](https://osteoporosiskneeclassification-4aetmz84zt7sfjydtmq4sb.streamlit.app)

📎 Sample X-ray images:
👉 [GitHub Sample Folder](https://github.com/PATRICK079/Osteoporosis_knee_Classification/tree/main/sample%20images)

📎 Video Demo of the App:
   [Video Demo](https://www.loom.com/share/5dfac25458be4e50b9955049390f6316?sid=2211fdc1-95d6-4dbe-9a0b-c63e847c9278)


# 🛠️ Technologies Used

•  Python

•  scikit-learn – for logistic regression

•  TensorFlow / Keras – for CNN image classification

•  Streamlit – app interface and deployment

•  Saturn Cloud – handled memory-heavy image augmentation

•  AWS S3 – hosted large CNN model files (>100MB)

## 🧠 Model Architecture

- **Custom CNN** with:
  - Convolutional layers
  - MaxPooling layers
  - Dense layers with dropout for regularization
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Activation Functions:** ReLU and Sigmoid

# ✅ Highlights

| Challenge                             | Solution                                                |
| ------------------------------------- | ------------------------------------------------------- |
| Small image dataset (372 images)      | Applied offline image augmentation to expand dataset    |
| Local memory issues during processing | Switched to Saturn Cloud for scalable resources         |
| GitHub file size limit (100MB)        | Deployed model via AWS S3 bucket integration            |
| Limited diagnostic flexibility        | Enabled both data-based and image-based diagnosis modes |

# 🧠 Key Takeaways
• **Full-cycle ML deployment:** From preprocessing and model training to cloud deployment.

• **Dual-modal ML app:** Blended structured data and image-based learning.

• **Real-world application:** Built to solve an actual medical need from a professional request.

• **Cloud integration experience:** Overcame hardware and GitHub limitations with cloud platforms.

# 📂 Project Structure

📦 Osteoporosis_knee_Classification/

├── app.py                  
├── logistic_model.pkl     
├── cnn_model.h5            
├── sample images/         
└── README.md               

# 👨‍⚕️ Use Case
This tool can support orthopedic clinics, general practitioners, and research teams by offering:

• Quick assessments based on patient data

• Visual X-ray classification when data is unavailable

• An accessible demo to validate results in real-time

# 📬 Contact
Patrick Edosoma

Machine Learning Engineer

[Linkedlin](https://www.linkedin.com/in/patrickedosoma/)

# ⭐️ Star This Repo
If you found this project helpful, please star ⭐️ it to show support!
