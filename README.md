# NYU-Data-Bootcamp-Final
## Diabetes Prediction Using Empirical and Machine Learning Models
Jiamei Shi(js14017), Siya Xu(sx2436), Kelly Yu(yy5089)
### 1. Introduction
With the increasing availability of large-scale health survey data, machine learning has become an important tool for early disease detection and risk prediction. Chronic diseases such as diabetes pose long-term health risks and significant economic costs, making early identification and prevention particularly valuable. Predictive modeling based on individual health indicators offers a scalable and data-driven approach to support public health decision-making.

In this project, we explore how effectively diabetes status can be predicted using structured health data. The focus is on comparing different modeling approaches, including classical statistical models, ensemble methods, and neural-network-based techniques, and understanding how they differ in predictive performance, their ability to handle class imbalance, and their level of interpretability.

By combining exploratory analysis with multiple predictive models and carefully chosen evaluation metrics, this project aims not only to assess how well diabetes can be predicted, but also to identify which modeling strategies are most suitable for imbalanced medical data in practice.

### 2. Research Objective
The central question of this project is:

How effectively can diabetes be predicted from health indicators, and how do different modeling approaches compare under class imbalance?

To address this question, the project does the following objectives:

Explore class structure and imbalance: Examine the distribution of diabetes outcomes and key health indicators.

Establish baseline performance: Use Logistic Regression and K-Nearest Neighbors as reference models.

Apply advanced machine learning methods: Implement Random Forest and XGBoost to capture nonlinear feature interactions.

Incorporate deep learning for tabular data: Use Multilayer Perceptron (MLP) and TabTransformer architectures.

Evaluate minority-class performance: Focus on recall, precision, and macro F1-score rather than accuracy alone.

Analyze trade-offs: Understand how model choice affects false negatives and false positives in a medical context.

### 3.Data Source
#### 3.1 Diabetes Health Indicators Dataset(Kaggle)
The dataset used in this project is derived from Kaggle named 2015 Behavioral Risk Factor Surveillance System (BRFSS), a large-scale health survey conducted annually in the United States. The dataset contains over 250,000 observations, each corresponding to an individual respondent.
The original target variable, Diabetes_012, takes three values:

0: No diabetes

1: Pre-diabetes

2: Diabetes

Depending on the model, the task is framed as either:
Binary classification, where prediabetes and diabetes are combined into a single positive class, or Multi-class classification, where all three categories are retained.

#### 3.2 Feature Description
The feature set includes a wide range of health-related indicators, such as:

Body Mass Index (BMI)

Physical health and mental health measures

Lifestyle behaviors (eg.smoking, exercise)

Demographic and medical history indicators

All features are numeric, though they differ in scale and interpretation. Some variables are continuous, while others are binary or ordinal.

#### 3.3 Preprocessing and Derived Variables
Several preprocessing steps were applied:

Creation of a binary diabetes indicator for selected models.

Train–test split with stratification to preserve class proportions.

Standardization of features for distance-based and linear models.

Separation of continuous and categorical features for TabTransformer.

Exploratory checks confirmed a strong class imbalance, motivating the use of class-weighted models, threshold tuning, and macro-averaged evaluation metrics.

### 4. Data Analysis

#### 4.1 Class Distribution
An initial analysis of the target variable revealed that the majority of observations fall into the non-diabetic class, with significantly fewer prediabetic and diabetic cases. This imbalance implies that predicting diabetes is inherently more difficult and that false negatives carry a higher cost.

#### 4.2 Feature Scaling and Structure
Summary statistics showed that features such as BMI and health-day counts exhibit substantial variation across individuals, while many lifestyle indicators are binary. This structure suggests that both linear models and nonlinear methods may be appropriate, depending on how interactions are captured.

#### 4.3 Implications for Modeling
The exploratory analysis highlights three challenges:
Class imbalance

Mixed feature types

Potential nonlinear interactions

These considerations guided the choice of models and evaluation strategies used in later stages.

### 5. Modeling Strategy
All models are organized into three parts.

#### 5.1 Part 1: Baseline Models
##### Logistic Regression
Logistic Regression serves as a baseline for both binary and multi-class classification. The model is trained on standardized features and optimized using maximum likelihood. Despite its simplicity, Logistic Regression provides strong interpretability and establishes a performance benchmark.

##### K-Nearest Neighbors (KNN)
KNN is implemented with k=5k=5k=5 using scaled features. As a distance-based method, KNN is sensitive to feature scaling and class imbalance. Results show that while KNN performs reasonably on the majority class, it struggles to identify diabetic cases.

#### 5.2 Part 2: Advanced Machine Learning Models
##### Random Forest
A Random Forest classifier with class_weight='balanced' is used to explicitly address class imbalance. By penalizing misclassification of minority-class observations, the model achieves substantially higher recall for diabetic cases, making it suitable for screening tasks.

##### XGBoost
XGBoost is implemented as a multi-class classifier using boosted decision trees. The model captures nonlinear relationships and interactions among health indicators and achieves strong overall accuracy and macro F1-score. It performs well on structured tabular data and achieves strong overall accuracy and macro F1-score, though its predictions are less transparent compared to simpler models.

#### 5.3 Part 3: Deep Learning Models
##### Multilayer Perceptron (MLP)
The MLP is a feed-forward neural network trained on standardized features. By default, it optimizes overall accuracy and tends to favor the majority class. To improve minority-class detection, the classification threshold is lowered, increasing recall at the expense of precision. While flexible, the model behaves largely as a black box.

##### TabTransformer
TabTransformer is a deep learning architecture designed specifically for tabular data. It embeds categorical variables and applies Transformer attention mechanisms to learn feature interactions. Continuous and categorical features are handled separately, allowing the model to capture complex dependencies(approximately 98%). However, this performance is largely driven by correct classification of the majority class.

#### 5.4 Part 4: Glivenko–Cantelli Empirical Bayes Classifier
While most predictive models in this project rely on parametric assumptions or optimization-based learning, an additional model was developed based on empirical distribution theory, motivated by the Glivenko–Cantelli (GC) theorem. This approach offers a transparent, probability-driven alternative to conventional machine learning classifiers.
##### 5.4.1 Theoretical Motivation
The Glivenko–Cantelli theorem states that the empirical cumulative distribution function (ECDF) converges uniformly to the true distribution function as the sample size grows. In large datasets such as BRFSS, this result implies that sample frequencies can be treated as consistent estimators of true probabilities.

Rather than fitting parameters via gradient-based optimization, the GC classifier directly estimates class-conditional probabilities from data and applies Bayes’ rule to perform classification. This approach bridges classical statistical theory and modern classification tasks, while avoiding strong parametric assumptions.

##### 5.4.2 Model Construction
The Glivenko–Cantelli classifier estimates diabetes risk by starting from the overall prevalence of diabetes in the training data and then adjusting this baseline using information from individual features. The baseline risk is computed directly from the sample proportion of diabetic cases and represents the average population-level likelihood of diabetes.

For each feature, the model compares how often a given value appears among diabetic versus healthy individuals. These probabilities are estimated directly from observed data rather than assumed distributions. Discrete variables use simple frequency counts, while continuous variables are grouped into quantile-based bins so they can be treated in the same probabilistic framework.

Each feature contributes a Weight of Evidence that indicates whether it increases or decreases diabetes risk. By summing these contributions with the baseline risk, the model produces a final probability estimate that remains transparent and easy to interpret, making it clear which features drive each prediction.

##### 5.4.3 Prediction and Interpretability
Predictions are generated by summing the base log-odds and all feature-level WoE contributions, followed by a sigmoid transformation to obtain probabilities.
A key advantage of this model is full interpretability. For any individual prediction, the classifier can display:

The baseline population risk

Each feature’s empirical contribution

Whether that feature increases or decreases predicted diabetes risk

##### 5.4.4 Threshold-Based Classification
As with other models in this project, a classification threshold is applied to convert predicted probabilities into binary outcomes. Using a threshold of 0.5, the GC classifier’s performance is evaluated alongside previously implemented models.

#### 6. Results and Interpretation 
Model performance is evaluated using recall, precision, F1-score, and accuracy, with particular emphasis on minority-class (diabetes) detection due to the imbalanced nature of the dataset.

The MLP model with a lowered classification threshold of 0.35 achieves a recall of 0.46 and a precision of 0.47, resulting in an F1-score of 0.46 and an accuracy of 0.83. This indicates a relatively balanced performance, with good overall accuracy but only moderate ability to identify diabetic cases.

The class-weighted Random Forest strongly prioritizes sensitivity to diabetes. It achieves the highest recall among all models at 0.76, meaning it successfully identifies most diabetic individuals. However, this comes at a significant cost to precision, which drops to 0.34, indicating a high rate of false positives. As a result, overall accuracy is reduced to 0.73, despite a slightly higher F1-score of 0.47.

The Glivenko–Cantelli (GC) classifier offers a middle ground between these two approaches. It achieves a recall of 0.52 and a precision of 0.41, yielding an F1-score of 0.46 and an accuracy of 0.81. Compared to the Random Forest, the GC model sacrifices some recall but substantially improves precision. Compared to the MLP, it improves recall while maintaining competitive accuracy, without requiring threshold tuning.

Overall, no single model dominates across all metrics. Instead, each model reflects a different trade-off between sensitivity, specificity, and overall predictive stability.

#### 7. Key Insights and Interpretation
First, class imbalance fundamentally shapes model behavior. Models that optimize for recall, such as the Random Forest, can substantially increase detection of diabetic cases but inevitably introduce many false positives. This highlights the limitation of relying on a single metric when evaluating medical classifiers.

Second, model complexity does not guarantee superior performance. Despite its simplicity, the MLP performs comparably to more advanced methods when threshold tuning is applied. Similarly, the GC classifier—despite avoiding parametric assumptions and iterative optimization—achieves performance on par with neural and ensemble models.

Third, interpretability and performance need not be mutually exclusive. While the Random Forest and MLP provide limited insight into feature-level contributions, the GC classifier maintains a transparent structure that allows each prediction to be decomposed into individual feature effects, while still achieving competitive recall, precision, and accuracy.

Finally, the results emphasize that the definition of “better” depends on the application context. In aggressive screening settings, maximizing recall may be preferred. In settings where false positives carry meaningful costs, a more balanced approach may be more appropriate.

#### 8. Conclusion
This project compared multiple machine learning and probabilistic approaches for predicting diabetes using large-scale health survey data. The results show that while ensemble methods such as Random Forest can achieve high sensitivity, they do so at the expense of precision and overall accuracy. Neural network models offer flexibility and strong accuracy but require threshold tuning and provide limited interpretability.

The Glivenko–Cantelli classifier demonstrates that a theory-driven, non-parametric approach can achieve competitive predictive performance while maintaining transparency and stability. Rather than outperforming all alternatives on a single metric, the GC model offers a balanced trade-off between recall and precision, along with the added benefit of clear, feature-level explanations.

Overall, the findings suggest that effective diabetes prediction is not solely a question of choosing the most complex model, but of selecting an approach that aligns with the intended use case, evaluation priorities, and interpretability requirements.

#### Reference
Kaggle Dataset: Diabetes Health Indicators Dataset, https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download



