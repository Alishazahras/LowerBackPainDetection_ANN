# Lower Back Pain Detection using Artificial Neural Network (ANN)
This project is aimed at building a machine learning model to assist in detecting whether a person has Lower Back Pain, utilizing an Artificial Neural Network (ANN) architecture. The dataset used for training, testing, and validation is titled `dataset_spline.csv`, provided by the company working in the healthcare sector. The following README outlines the steps taken in the project, from data preprocessing to model evaluation, along with the results.

## Project Objectives
1. **Data Preprocessing and Exploration:**
   - Identify and solve potential issues in the dataset, such as missing values, outliers, or improper data types.
   - Perform Exploratory Data Analysis (EDA) to better understand the data and identify key features influencing the diagnosis of lower back pain.
2. **Dataset Split:**

   Split the dataset into three parts with the following ratio:
      - 80% Training set
      - 10% Testing set
      - 10% Validation set
3. **ANN Architecture Design:**
   - Create an ANN model with the following architecture:
        - Input Layer
        - Hidden Layer 1: 512 neurons, Sigmoid activation
        - Hidden Layer 2: 256 neurons (N/2 of previous layer), Sigmoid activation
        - Hidden Layer 3: 128 neurons (N/2 of previous layer), Sigmoid activation
        - Output Layer: Softmax activation function
    - Visualize and analyze the training and validation loss over epochs.
4. **Model Optimization:**

    Modify the initial ANN architecture to optimize accuracy by tuning hyperparameters, adjusting the number of layers or neurons, and other techniques to achieve better performance.
5. **Model Evaluation:**
      - Evaluate the performance of both the initial and optimized models using the following metrics on the test set:
          - Accuracy
          - Precision
          - Recall
          - F1-Score
      - Provide a detailed explanation and interpretation of the results.
6. **Code Walkthrough and Explanation:**
    - Record a video explaining the code implementation, including how the data was processed, the architecture of the ANN, and the model's evaluation results.
    - Share insights and opinions regarding the model performance and evaluation.

## Dataset Description
The dataset contains patient-related features, which will be used to predict whether the patient suffers from lower back pain. The specific contents of the dataset (features, target labels, etc.) will be explored and cleaned in the data preprocessing step.

## Project Workflow
1. **Data Preprocessing:**
    - Clean and preprocess the dataset, handling missing values, outliers, or incorrect data types.
    - Use appropriate scaling techniques if required.
2. **Exploratory Data Analysis (EDA):**
    - Perform data visualization to understand distributions, correlations, and relationships among variables.
    - Identify any imbalances in the dataset that might affect model performance.
3. **Model Building:**
    - Implement the initial ANN model with 3 hidden layers as described above.
    - Train the model and track performance metrics such as training loss, validation loss, accuracy, etc.
4. **Model Optimization:**
    - Fine-tune the architecture and hyperparameters of the ANN (such as learning rate, batch size, activation functions, etc.) to enhance model performance.
    - Experiment with different techniques and architectures to improve accuracy and reduce overfitting or underfitting.
5. **Model Evaluation:**
    - Use evaluation metrics to assess model performance and compare the results of the initial and optimized models.
    - Provide detailed insights into accuracy, precision, recall, and F1-score results.
6. **Video Explanation:**
    - Record and provide a video explaining the entire process, code, and results.

## Requirements
- Python 3.x
- Libraries:
  - `TensorFlow`
  - `Pandas`
  - `NumPy`
  - `Scikit-learn`
  - `Matplotlib`
  - `Seaborn`
 
## Results
- Initial ANN Model: The initial architecture was built with three hidden layers, as specified. Results of the accuracy and other metrics will be detailed in the report.
- Optimized ANN Model: After hyperparameter tuning, architecture modifications, and experimentation, the optimized model achieved a better performance than the initial model.

## Conclusion
This project provides a comprehensive approach to building and optimizing an ANN model to detect lower back pain using a provided dataset. Through a structured approach in data preprocessing, model building, and optimization, the best possible accuracy for this task was obtained.
