# Kaiburr Technical Assessment - Task 5: Data Science - Consumer Complaint Classification

This repository contains the solution for **Task 5** of the Kaiburr technical assessment. It is a complete data science project that builds and evaluates a machine learning model to classify consumer financial complaints. The project includes comprehensive exploratory data analysis (EDA), text pre-processing, and a comparative analysis of five different classification models.

---

## Table of Contents
1. [Project Objective](#1-project-objective)
2. [Dataset Selection](#2-dataset-selection)
3. [Project Workflow](#3-project-workflow)
4. [Results and Performance](#4-results-and-performance)
5. [How to Run](#5-how-to-run)
6. [Key Visualizations (Screenshots)](#6-key-visualizations-screenshots)

---

### 1. Project Objective
The goal of this project is to develop a robust text classification model that can accurately categorize consumer complaints into four predefined target categories:
- Credit reporting/other
- Debt collection
- Consumer Loan
- Mortgage

---

### 2. Dataset Selection
The dataset suggested in the task description was the official "Consumer Complaint Database" from data.gov, which is over 7 GB in size. Due to computational constraints that made processing a file of this magnitude impractical on a local machine, a smaller but highly relevant alternative was chosen from Kaggle.

- **Dataset Used:** [Consumer Complaint Database (Kaggle)](https://www.kaggle.com/datasets/selener/consumer-complaint-database)
- **Size:** ~800 MB
- **Justification:** This Kaggle dataset is a large, recent snapshot of the same official data source. It is substantial enough to build a robust, high-performing model while remaining manageable for development and training. **All the required tasks and analysis steps outlined in the assignment were applied to this dataset.**
- **File Used in this Repo:** `rows.csv`

---

### 3. Project Workflow
The project follows a standard data science workflow:
1.  **Data Loading & EDA:** The dataset was loaded, and an in-depth exploratory data analysis was performed to understand data distributions, missing values, and key textual patterns using heatmaps, bar charts, pie charts, and word clouds.
2.  **Text Pre-processing:** Complaint narratives were cleaned and standardized using tokenization, stop word removal, and lemmatization to prepare them for feature extraction.
3.  **Feature Engineering:** A numerical target variable was created, and the cleaned text was vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) technique, transforming text into a numerical format suitable for machine learning.
4.  **Model Training & Comparison:** Five different classification models were trained and evaluated to find the best performer:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Support Vector Machine (LinearSVC)
    - LightGBM (Gradient Boosting)
    - MLP (Multi-layer Perceptron Neural Network)
5.  **Model Evaluation:** Models were rigorously compared based on accuracy and training time to select the optimal solution.

---

### 4. Results and Performance
After a comprehensive comparison, the **MLP (Neural Network) model** was identified as the top performer, showcasing the power of neural architectures on large, complex text data.

#### Model Comparison Summary

| Model | Accuracy (%) | Training Time (s) |
| :--- | :--- | :--- |
| **MLP (Neural Network)** | **88.27%** | 479.98 |
| LightGBM | 87.85% | 678.82 |
| Logistic Regression | 86.61% | 50.37 |
| Support Vector Machine (LinearSVC) | 86.49% | 76.78 |
| Multinomial Naive Bayes | 83.63% | 0.12 |

<br>

#### Detailed Classification Report for the Best Model (MLP)

The winning model achieved an overall accuracy of **88.27%** and demonstrated strong, balanced performance across all four categories.

| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Credit reporting/other | 0.89 | 0.89 | 0.89 | 21052 |
| Debt collection | 0.87 | 0.88 | 0.88 | 17247 |
| Consumer Loan | 0.84 | 0.80 | 0.82 | 9512 |
| Mortgage | 0.93 | 0.94 | 0.94 | 10534 |
| **---** | **---** | **---** | **---** | **---** |
| **Accuracy** | | | **0.88** | **58345** |
| **Macro Avg** | 0.88 | 0.88 | 0.88 | 58345 |
| **Weighted Avg** | 0.88 | 0.88 | 0.88 | 58345 |

---

### 5. How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ashraf0705/kaiburr-task5-Data-Science.git
    cd kaiburr-task5-Data-Science
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk lightgbm wordcloud jupyter
    ```
3.  **Launch and run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Consumer_Complaint_Classification.ipynb"
    ```
    *(Note: Rename your notebook file to this or update the command accordingly)*

    Execute the cells in the notebook sequentially to replicate the analysis.

---

### 6. Key Visualizations (Screenshots)

*Each screenshot includes the date/time and author's name as required.*

#### EDA: Distribution of Complaint Categories
This plot from the EDA phase highlights the class imbalance in the dataset.

<p align="center">
  <img src="./screenshots/distribution.png" alt="Distribution of Complaint Categories" width="900" />
</p>

---

#### EDA: Geographic Distribution of Complaints
This bar plot shows the top 20 states by complaint volume.

<p align="center">
  <img src="./screenshots/bar_plot.png" alt="Top 20 States by Complaint Volume" width="900" />
</p>

---

#### Final Model Performance Comparison
This visualization summarizes the accuracy and training time for all five models, clearly showing the MLP as the most accurate.

<p align="center">
  <img src="./screenshots/model_accuracy.png" alt="Model Accuracy and Training Time Comparison" width="900" />
</p>
