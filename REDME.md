# Heart Disease Prediction Project

## Contents
1. Introduction
2. Data Collection
3. Data Preprocessing
4. Exploratory Data Analysis
5. Model Selection
6. Model Training and Evaluation
7. Model Deployment
8. Conclusion
9. References

## Introduction
Heart disease is one of the leading causes of death worldwide. Early prediction and diagnosis can significantly improve patient outcomes. This project aims to develop a machine learning model to predict the likelihood of heart disease based on various medical attributes.

## Data Collection
The dataset used in this project is a CSV file named `student_version.csv`. It contains 734 instances and 12 attributes, including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, oldpeak, ST slope, and heart disease status.

## Data Preprocessing
### Handling Missing Values
- Missing values were handled using imputation techniques. For numerical features, missing values were replaced with the mean of the column. For categorical features, the most frequent value was used.

### Encoding Categorical Variables
- Categorical variables were converted into numerical values using `LabelEncoder` and `OneHotEncoder`.

### Scaling
- Numerical features were standardized to bring them to a similar scale.

## Exploratory Data Analysis
### Key Observations
1. **Chest Pain Type**: Different types of chest pain may correlate with the likelihood of heart disease.
2. **Resting Blood Pressure**: Higher resting blood pressure may be a significant indicator of heart disease risk.
3. **Cholesterol**: Elevated cholesterol levels can be a strong risk factor for heart disease.
4. **Fasting Blood Sugar**: Elevated fasting blood sugar could be linked to increased heart disease risk.
5. **Resting ECG**: Abnormal ECG readings may be associated with a higher likelihood of heart disease.
6. **Max Heart Rate**: Lower maximum heart rate responses during stress tests could indicate underlying cardiac issues.
7. **Exercise Angina**: The presence of exercise-induced angina is a critical indicator of heart disease risk.
8. **Oldpeak**: The oldpeak measurement can be a valuable predictor of heart disease.
9. **ST Slope**: The slope of the ST segment during exercise tests can serve as an important diagnostic criterion.

## Model Selection
Several machine learning models were considered for this classification task, including:
1. **Random Forest Classifier**
2. **Logistic Regression**
3. **Neural Network**
4. **K-Neighbors Classifier**

## Model Training and Evaluation
### Cross-Validation
The dataset was split into training, testing, and validation sets to ensure reliable performance metrics.

### Model Performance
1. **Random Forest Classifier**
   - Cross-Validation Accuracy: 89.12%
   - Test Accuracy: 88.44%

2. **Logistic Regression**
   - Validation Accuracy: 85.71%
   - Test Accuracy: 89.12%

3. **Neural Network**
   - Validation Accuracy: 85.03%
   - Test Accuracy: 87.07%

4. **K-Neighbors Classifier**
   - Validation Accuracy: 82.31%
   - Test Accuracy: 80.95%

### Conclusion
Both Logistic Regression and Random Forest Classifier were chosen due to their superior accuracy compared to other models.

## Model Deployment
The trained model was deployed using Docker and Azure. The deployment process involved containerizing the application and hosting it on Azure for scalability and accessibility.

### Docker Deployment
```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app_ml

# Copy the current directory contents into the container
COPY . /app_ml

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=run.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
```

### Azure Deployment
The Docker container was deployed on Azure, ensuring that the model is accessible via a web API for real-time predictions.

## Conclusion
This project successfully developed a machine learning model to predict heart disease. The model was trained and evaluated on the provided dataset and deployed using Docker and Azure for real-time predictions.

