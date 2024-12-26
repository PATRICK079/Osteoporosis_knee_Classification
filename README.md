# Osteoporosis_knee_Classification
![man-knee-pain-he-puts-260nw-2476578973 jpg](https://github.com/user-attachments/assets/69d8db39-531e-417d-a6a5-79578d115f0a)

# Introduction
This project aims to predict whether a patient has a healthy knee or one affected by osteoporosis. By using advanced machine learning techniques, the application offers two prediction modes to assess knee health.

The focus of this project is to develop a user-friendly tool that helps identify individuals at risk of osteoporosis, a condition that weakens bone density and increases the likelihood of fractures. Osteoporosis often leads to mobility issues and other health complications. The application combines data analysis and prediction models to provide valuable insights into an individual’s bone health, offering a clearer picture of their risk for osteoporosis-related knee problems.
# Business Statement
As a machine learning expert in a healthcare organization, one of the orthopedic doctors has approached me to leverage my expertise. He has requested that I develop a machine learning model capable of identifying healthy knees and those affected by osteoporosis using knee X-rays and patient data. The goal is to enhance and expedite his decision-making process. Additionally, the models will be deployed using Streamlit to ensure a user-friendly interface for practical application.

# Dataset Dictionary
This app is designed to predict whether a patient has a healthy knee or an osteoporosis-affected knee. Leveraging cutting-edge machine learning techniques, the app offers two prediction modes:

1. Patient Data Analysis: Input patient data to get a prediction based on a Logistic Regression Model.

• Joint Pain: Indicates whether the individual experiences joint pain, with options "Yes" or "No."

• Gender: Gender of the individual, with options "Male" or "Female."

• Age: The individual's age in years, ranging from 0 to 120 years.

• Menopause Age: The age at which menopause occurred (applicable only for females), ranging from 0.0 to 100.0 years. Male = 0

• Height (meters): The height of the individual in meters, ranging from 0.0 to 3.0 meters.

• Weight (kg): The weight of the individual in kilograms, ranging from 0.0 to 300.0 kg.

• Smoker: Indicates whether the individual smokes, with options "Yes" or "No."

• Diabetic: Indicates whether the individual has diabetes, with options "Yes" or "No."

• Hypothyroidism: Indicates whether the individual has hypothyroidism, with options "Yes" or "No."

• Number of Children: The total number of pregnancies or children the individual has had, ranging from 0 to 20.

• Seizure Disorder: Indicates if the individual has a seizure disorder, with options "Yes" or "No."

• Estrogen Use: Indicates whether the individual uses estrogen (applicable only for females), with options "Yes" or "No."

• History of Fracture: Indicates if the individual has a history of fractures, with options "Yes" or "No."

• Dialysis: Indicates whether the individual is undergoing dialysis, with options "Yes" or "No."

• Family History of Osteoporosis: Indicates if there is a family history of osteoporosis, with options "Yes" or "No."

• Maximum Walking Distance (km): Maximum distance the individual can walk without significant discomfort, ranging from 0.0 to 10.0 km.

• Medical History: Indicates whether the individual has any notable medical history, with options "Yes" or "No."

• T-Score Value: The T-score value used for bone density analysis, ranging from -20.0 to 10.0.

• Z-Score Value: The Z-score value used for bone density analysis, ranging from -20.0 to 10.0.

• BMI (Body Mass Index): Body Mass Index of the individual, ranging from 0.0 to 50.0.

• Obesity: Indicates whether the individual is considered obese, with options "Yes" or "No."

2. Knee X-ray Image Analysis: Upload a knee X-ray image to utilize a Convolutional Neural Network (CNN) model trained on the Osteoporosis Knee X-ray Dataset from Kaggle. Incorporates offline image augmentation for enhanced accuracy and robustness.

# Tools used 
1. Tensorflow
2. AWS S3 bucket
3. Git
4. Saturn cloud
5. Streamlit
6. python
7. scikit-learn


# Problems encountered in this project and solutions

1. Insufficient Dataset for CNN Model: The initial dataset of only 372 images was too small for training a CNN model effectively. To address this, I performed offline image augmentation, generating new transformed images and expanding the dataset.

2. Memory Issues During Image Augmentation: Saving the augmented images overwhelmed the memory on my local machine. To resolve this, I transitioned to Saturn Cloud, which provided the necessary resources for handling the dataset efficiently.

3. GitHub Upload Limit for Model: The CNN model file was about 290 MB, which exceeded GitHub's 100 MB upload limit. I overcame this by uploading the model to an AWS S3 bucket, enabling me to deploy it via Streamlit.

# Key Learnings from the Project 
Throughout the process of building, training, and deploying the CNN model for knee image classification, I gained valuable insights and practical experience in several key areas:

1. Data Augmentation Techniques: I learned the importance of data augmentation when working with small datasets. By generating additional images through transformations, I was able to effectively increase the dataset size, which improved the performance and generalization of the model.
   
2. Cloud Computing for Resource Management: Moving my project to Saturn Cloud taught me how to leverage cloud computing resources to overcome local machine limitations. This experience showed me the value of using cloud platforms for handling large datasets and running resource-intensive processes without running into memory constraints.
   
3. Model Deployment Challenges: I encountered practical challenges related to file size limitations when deploying models. By using AWS S3 for storing large model files, I learned how to work around restrictions like GitHub's file size limit and gained experience in integrating cloud storage with deployment tools like Streamlit.
 
4. Adaptability and Problem-Solving: The project required me to think creatively and quickly adapt to overcome technical hurdles. These experiences strengthened my problem-solving skills and deepened my understanding of how to handle real-world challenges in machine learning and model deployment.


