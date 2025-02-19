# Principal Component Analysis (PCA) with Logistic Regression

## Project Overview
This project demonstrates the application of **Principal Component Analysis (PCA)** for dimensionality reduction and **Logistic Regression** for classification using the `Wine.csv` dataset. The dataset is first preprocessed, reduced to two principal components using PCA, and then classified using a logistic regression model. The results are visualized to understand the decision boundaries.

## Dataset
The dataset `Wine.csv` contains multiple features describing different types of wine. The last column represents the class label (wine type). PCA is applied to reduce the feature dimensions while preserving the most important variance in the data.

## Installation & Requirements
Ensure you have Python installed along with the following dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Implementation Steps
1. **Import Required Libraries**
   - `numpy`, `pandas` for data handling
   - `matplotlib` for visualization
   - `scikit-learn` for PCA, logistic regression, and model evaluation

2. **Load Dataset**
   - The dataset is loaded using Pandas and split into features (`X`) and target variable (`y`).

3. **Split Data into Training and Testing Sets**
   - `train_test_split` is used to divide the dataset into 80% training and 20% testing data.

4. **Feature Scaling**
   - Standardization is applied using `StandardScaler` to normalize the features before applying PCA.

5. **Apply PCA for Dimensionality Reduction**
   - PCA is performed to reduce feature dimensions to `n_components=2`.
   - The explained variance ratio is printed to show how much variance each principal component captures.

6. **Train Logistic Regression Model**
   - A logistic regression classifier is trained on the transformed dataset.

7. **Evaluate Model Performance**
   - Predictions are made on the test set.
   - Performance is measured using a **confusion matrix** and **accuracy score**.

8. **Visualization of Decision Boundaries**
   - The decision boundary of the logistic regression model is plotted using `contourf` to see how PCA transformed data is classified.
   - Training and test set decision boundaries are visualized separately.

## Usage
To run the project, execute the Python script or Jupyter Notebook step by step:

```bash
python principal_component_analysis.py
```

## Results & Interpretation
- PCA helps in reducing the number of features while retaining most of the important information.
- Logistic Regression performs well in classification using just **two principal components**.
- The **decision boundary visualization** provides insight into how PCA-transformed data is classified.

## Future Improvements
- Experiment with different values of `n_components` to see how it affects classification accuracy.
- Try other classifiers (e.g., **SVM, Random Forest**) to compare performance.
- Apply PCA to larger, high-dimensional datasets for better visualization and computational efficiency.

## Author
Developed by **Ahmad Alsebai**

## License
This project is open-source and free to use for educational and research purposes.

