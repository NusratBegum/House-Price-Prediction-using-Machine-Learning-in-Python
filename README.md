# House Price Prediction using Machine Learning in Python

## Project Description

This project implements a comprehensive machine learning solution for predicting house prices using the Boston Housing dataset. The analysis demonstrates the complete data science workflow, from exploratory data analysis through model development and evaluation. By comparing Simple Linear Regression with Multiple Linear Regression approaches, the project illustrates how incorporating multiple features can significantly improve prediction accuracy.

The Boston Housing dataset contains information collected by the U.S. Census Service concerning housing in the Boston, Massachusetts area. This project serves as both a practical application of regression techniques and an educational resource for understanding machine learning fundamentals.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Comprehensive Exploratory Data Analysis (EDA)**: Detailed statistical analysis and visualization of the dataset
- **Data Quality Assessment**: Missing value analysis, duplicate detection, and outlier identification
- **Statistical Hypothesis Testing**: Formal validation of relationships between features and target variable
- **Feature Engineering**: Creation of derived features to enhance model performance
- **Multiple Model Comparison**: Implementation and comparison of Simple and Multiple Linear Regression
- **Rigorous Model Evaluation**: Assessment using R² Score, RMSE, MAE, and cross-validation
- **Professional Documentation**: Clear, well-commented code with detailed explanations

## Dataset

### Boston Housing Dataset

- **Source**: UCI Machine Learning Repository / U.S. Census Service
- **Size**: 506 observations
- **Features**: 13 predictor variables
- **Target Variable**: Median value of owner-occupied homes (MEDV) in $1000s

### Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| CRIM | Per capita crime rate by town | Continuous |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft. | Continuous |
| INDUS | Proportion of non-retail business acres per town | Continuous |
| CHAS | Charles River dummy variable (1 if bounds river; 0 otherwise) | Binary |
| NOX | Nitric oxides concentration (parts per 10 million) | Continuous |
| RM | Average number of rooms per dwelling | Continuous |
| AGE | Proportion of owner-occupied units built prior to 1940 | Continuous |
| DIS | Weighted distances to five Boston employment centres | Continuous |
| RAD | Index of accessibility to radial highways | Discrete |
| TAX | Full-value property-tax rate per $10,000 | Continuous |
| PTRATIO | Pupil-teacher ratio by town | Continuous |
| B | 1000(Bk - 0.63)^2 where Bk is proportion of Black residents | Continuous |
| LSTAT | Percentage of lower status population | Continuous |
| MEDV | Median value of owner-occupied homes in $1000s | Continuous (Target) |

## Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/NusratBegum/House-Price-Prediction-using-Machine-Learning-in-Python.git
   ```

2. Navigate to the project directory:
   ```bash
   cd House-Price-Prediction-using-Machine-Learning-in-Python
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install individual packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Open `index.ipynb` in your browser

## Usage

### Running the Analysis

1. Open the Jupyter Notebook (`index.ipynb`)
2. Execute cells sequentially from top to bottom using `Shift + Enter`
3. The notebook is organized into clear sections for easy navigation
4. Modify parameters and experiment with different features as needed

### Key Sections

- **Data Loading**: Imports the Boston Housing dataset from an online source
- **EDA**: Visualizations and statistical summaries of all features
- **Preprocessing**: Data cleaning and outlier handling
- **Modeling**: Training Simple and Multiple Linear Regression models
- **Evaluation**: Comprehensive performance metrics and comparison

### Customization

To use your own dataset:
1. Replace the data loading section with your data source
2. Ensure your dataset has appropriate features and a continuous target variable
3. Adjust feature names and descriptions accordingly
4. Modify visualizations to match your data characteristics

## Project Structure

```
House-Price-Prediction-using-Machine-Learning-in-Python/
│
├── index.ipynb                 # Main Jupyter notebook containing all analysis
├── README.md                   # Project documentation (this file)
└── .git/                       # Git version control directory
```

## Methodology

### 1. Data Understanding
- Load and inspect the dataset
- Analyze feature types and distributions
- Create comprehensive feature documentation

### 2. Data Preprocessing
- Check for missing values
- Detect and handle duplicates
- Identify outliers using IQR method
- Decide on outlier treatment strategy

### 3. Exploratory Data Analysis
- Target variable distribution analysis
- Univariate analysis of each feature
- Correlation analysis between features
- Bivariate analysis (features vs target)

### 4. Statistical Testing
- Test normality assumptions
- Validate feature-target relationships
- Perform hypothesis tests to confirm patterns

### 5. Feature Engineering
- Create derived features based on domain knowledge
- Combine related features
- Transform variables to improve model performance

### 6. Model Development
- **Simple Linear Regression**: Single feature (RM - average rooms)
- **Multiple Linear Regression**: All relevant features
- Split data into training and testing sets (80/20)
- Train models using scikit-learn

### 7. Model Evaluation
- Calculate R² Score, RMSE, and MAE
- Perform k-fold cross-validation
- Compare model performance
- Generate prediction examples

## Results

### Model Performance Comparison

The Multiple Linear Regression model demonstrates superior performance compared to the Simple Linear Regression model:

#### Simple Linear Regression (Single Feature: RM)
- Uses only the average number of rooms as predictor
- Demonstrates basic linear relationship
- Baseline model for comparison

#### Multiple Linear Regression (All Features)
- Incorporates all 13 predictor variables
- Significantly improved prediction accuracy
- Better captures complex relationships in the data

### Key Findings

1. **RM (Average Rooms)** shows strong positive correlation with house prices
2. **LSTAT (Lower Status Population %)** shows strong negative correlation
3. Multiple features together provide much better predictions than single feature
4. Statistical hypothesis tests confirm significant relationships between features and prices
5. Model cross-validation shows consistent performance across different data splits

### Insights

- Properties with more rooms command higher prices
- Socioeconomic factors (LSTAT, PTRATIO) significantly impact valuations
- Environmental factors (NOX, CRIM) negatively affect property values
- Location features (CHAS, DIS) contribute to price variations
- Multiple Linear Regression captures these complex interactions effectively

## Technologies Used

### Core Technologies
- **Python 3.x**: Primary programming language
- **Jupyter Notebook**: Interactive development environment

### Libraries and Frameworks

#### Data Manipulation
- **Pandas**: Data structures and analysis tools
- **NumPy**: Numerical computing and array operations

#### Visualization
- **Matplotlib**: Basic plotting and charts
- **Seaborn**: Statistical visualizations and themes

#### Machine Learning
- **scikit-learn**: Machine learning models and evaluation metrics
  - `LinearRegression`: Regression model implementation
  - `train_test_split`: Data splitting
  - `cross_val_score`: Cross-validation
  - `StandardScaler`: Feature scaling
  - Performance metrics (MSE, MAE, R²)

#### Statistical Analysis
- **SciPy**: Statistical tests and functions
  - Hypothesis testing (t-tests, ANOVA)
  - Normality tests (Shapiro-Wilk)
  - Correlation tests (Pearson, Spearman)

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/YourFeature`
3. **Make your changes**: Improve code, fix bugs, or add features
4. **Commit your changes**: `git commit -m 'Add some feature'`
5. **Push to the branch**: `git push origin feature/YourFeature`
6. **Open a Pull Request**: Describe your changes and their benefits

### Contribution Guidelines
- Maintain consistent code style
- Add comments for complex logic
- Update documentation for new features
- Test your changes thoroughly
- Follow Python PEP 8 style guide

## License

This project is available as open source. Feel free to use, modify, and distribute the code for educational and commercial purposes.

## Acknowledgments

- **Dataset Source**: Boston Housing Dataset from the UCI Machine Learning Repository, originally collected by the U.S. Census Service
- **Inspiration**: Classic machine learning regression problem for educational purposes
- **Community**: Thanks to the data science community for tutorials and best practices
- **scikit-learn**: Excellent documentation and implementation of machine learning algorithms
- **Jupyter Project**: For providing an outstanding interactive computing environment

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the GitHub repository.

## Project Status

This project is complete as a demonstration of linear regression techniques. Future enhancements could include:
- Implementation of advanced regression techniques (Ridge, Lasso, ElasticNet)
- Feature selection using recursive feature elimination
- Ensemble methods (Random Forest, Gradient Boosting)
- Deep learning approaches
- Deployment as a web application
- Real-time prediction API

---

**Note**: The Boston Housing dataset contains a variable (B) that may be considered problematic from an ethical standpoint as it relates to racial demographics. This project uses the dataset purely for educational purposes to demonstrate machine learning techniques. Modern practitioners should be aware of ethical considerations in feature selection and model development.
