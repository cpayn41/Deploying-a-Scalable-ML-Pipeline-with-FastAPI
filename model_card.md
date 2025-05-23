# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Creator**: Christina Payne
- **Created**: May 2025
- **Model Type**: Random Forest Classifier
- **Framework**: scikit-learn
- **Version**: 1.0

## Intended Use
- **Primary Use**: Predict whether an individual earns more than $50K annually based on demographic features.
- **Users**: Data scientists, analysts studying income distribution.
- **Out of Scope**: Real-time predictions or critical decision-making without validation.

## Factors
- **Features**: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.
- **Target**: salary (<=50K or >50K).
- **Data Source**: Census Bureau data (census.csv).

## Training Data
- **Dataset**: 80% training split from census.csv (~26,049 rows).
- **Preprocessing**: Same as evaluation data.

## Evaluation Data
- **Dataset**: 20% test split from census.csv (~6,512 rows).
- **Preprocessing**: One-hot encoding for categorical features, numerical features unchanged.

## Metrics
Precision, Recall, F1 Score.
- **Overall Performance** (on test set):
  - Precision: 0.7391
  - Recall: 0.6384
  - F1 Score: 0.6851
- **Slice Performance**: See slice_output.txt for metrics by categorical feature (e.g., workclass: Private, F1: 0.6838; sex: Male, F1: 0.6985; Female, F1: 0.5995).
## Quantitative Analyses
- Model performs better for some slices (e.g., workclass: Private, F1: 0.6838) than others (e.g., workclass: Without-pay, F1: 1.0000, Count: 4, low sample size). See slice_output.txt for details.

## Ethical Considerations
- **Bias**: Model may reflect biases in data (e.g., race, sex). Slice metrics show varying performance (e.g., sex: Male, F1: 0.6985; Female, F1: 0.5995).
- **Fairness**: Avoid using predictions to discriminate against protected groups.
- **Transparency**: Model card and slice metrics provided.

## Caveats and Recommendations
- **Limitations**: Low sample sizes for some slices (e.g., native-country: Yugoslavia, Count: 2) may lead to unreliable metrics.
- **Recommendations**: Validate on diverse datasets, consider fairness-aware algorithms to mitigate bias.

