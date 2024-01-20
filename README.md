Spam SMS Classification Project Overview:
This project revolves around the classification of SMS messages into spam or ham (non-spam) categories. The dataset 'spam.csv' is utilized for training and evaluating a machine learning model to distinguish between the two classes.

Data Loading and Exploration:
The dataset is loaded, and initial exploration involves extracting relevant columns ('v1', 'v2') and renaming them to ('label', 'message'). The 'label' column is encoded into binary values ('ham': 0, 'spam': 1), and any missing values in the 'label' column are removed.

Label Distribution Visualization:
A countplot is generated to visualize the distribution of labels ('ham' and 'spam') in the dataset, providing an overview of the class distribution.

Data Splitting:
The dataset is split into training and testing sets using the 'train_test_split' function, allocating 20% of the data for testing and ensuring reproducibility with a random state of 42.

Text Vectorization:
The 'message' column is transformed into TF-IDF vectors using the 'TfidfVectorizer' with English stop words and a maximum of 5000 features.

Naive Bayes Classification:
A Multinomial Naive Bayes classifier is trained on the TF-IDF vectors for spam classification.

Model Evaluation:
The classifier is evaluated using a confusion matrix heatmap, illustrating true positive, true negative, false positive, and false negative predictions. Additional evaluation metrics such as accuracy, precision, recall, and F1-score are computed and presented.

Message Length Distribution Visualization:
A histogram is generated to showcase the distribution of message lengths for both spam and ham messages.

Cross-Validation Scores:
Cross-validation scores are calculated using the Multinomial Naive Bayes model to assess its generalization performance.

Sensitivity to Hyperparameter Alpha:
The project explores the impact of different alpha values (0.1, 1.0, 10.0) on the Multinomial Naive Bayes model's accuracy through a loop, providing insights into the model's sensitivity to alpha changes.
