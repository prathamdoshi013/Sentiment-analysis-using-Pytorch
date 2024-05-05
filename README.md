# Sentiment-analysis-using-Pytorch

## BERT Sentiment Analysis

This project implements sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers) with PyTorch. The model is trained and evaluated on a sentiment classification dataset.

### Requirements
- pandas
- numpy
- torch
- transformers
- scikit-learn

### Installation
1. Install the required packages using pip:
   ```
   pip install pandas numpy torch transformers scikit-learn
   ```
2. Clone the repository:
   ```
   git clone https://github.com/your-username/bert-sentiment-analysis.git
   ```
3. Navigate to the project directory:
   ```
   cd bert-sentiment-analysis
   ```

### Usage
1. Load the sentiment classification dataset using Pandas.
2. Split the dataset into training, validation, and test sets.
3. Tokenize and encode sequences using the BERT tokenizer.
4. Define the BERT model architecture for sentiment classification.
5. Train the model using the training dataset.
6. Evaluate the model using the validation dataset.
7. Test the model on the test dataset and generate classification reports.
8. Save the trained model.
