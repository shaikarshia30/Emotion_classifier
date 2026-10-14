# Emotion-Aware Overlapping Community Detection in Social Networks

## Project Overview

This project aims to combine fine-grained emotion classification with
social-network analysis to study the emotional composition of
overlapping communities.

The current notebook, `emotion_classifier`, focuses on the initial
research pipeline: 1. Load the GoEmotions dataset. 2. Download and load
a Reddit conversation corpus through ConvoKit. 3. Clean and save the
datasets. 4. Fine-tune a DistilBERT model for multi-label emotion
classification. 5. Evaluate the classifier and save emotion
probabilities for the test set.

**Note:** The notebook currently covers dataset preparation and emotion
classification. Graph construction, overlapping community detection,
temporal analysis, and the web dashboard are planned extensions and
should not be considered implemented until their code and results are
added.

## Objectives

-   Classify text into fine-grained emotion categories.
-   Prepare conversation data for social-network analysis.
-   Save reproducible datasets, model outputs, and evaluation metrics.
-   Use the predicted emotions as a later input to user-level and
    community-level analysis.

## Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Research environment    Google Colab / Jupyter  Develop and run the
                          Notebook                research pipeline

  Data processing         Python, Pandas, NumPy   Load, clean, and
                                                  transform datasets

  Conversation data       ConvoKit                Download and process a
                                                  Reddit conversation
                                                  corpus

  Emotion dataset         GoEmotions              Train and evaluate the
                                                  emotion classifier

  NLP model               DistilBERT via Hugging  Predict multiple
                          Face Transformers       emotion labels per text

  Model training          PyTorch, Hugging Face   Fine-tune and evaluate
                          Trainer                 the model

  Evaluation              scikit-learn            Calculate F1 scores and
                                                  accuracy

  Output storage          CSV and JSON            Store processed data,
                                                  predictions, and
                                                  cleaning reports
  -----------------------------------------------------------------------

## Datasets

### 1. GoEmotions

GoEmotions is a fine-grained emotion dataset containing Reddit comments
annotated with 27 emotion categories plus Neutral. The notebook loads
the official training, validation, and test splits.

Dataset source:
https://github.com/google-research/google-research/tree/master/goemotions

The notebook uses these splits: - `train.tsv` - `dev.tsv` as the
validation split - `test.tsv`

### 2. ConvoKit Reddit Corpus

The notebook downloads the `reddit-corpus-small` corpus using ConvoKit
and extracts conversation-level fields such as utterance ID, text,
speaker ID, reply target, conversation ID, and timestamp where
available.

ConvoKit: https://convokit.cornell.edu/

**Important data limitation:** Do not assume that speaker identities in
the ConvoKit corpus match comment IDs or authors in GoEmotions.
GoEmotions is used to train the text emotion classifier; the classifier
can then be applied to conversation messages. The interaction graph
should use identities and reply relationships from the conversation
corpus itself.

## Prerequisites

-   A Google account if running in Google Colab.
-   Internet access to download datasets and pretrained model files.
-   A runtime with sufficient memory. A GPU is recommended for model
    fine-tuning, but training speed depends on the available hardware.

## How to Run

1.  Upload or open the `emotion_classifier` notebook in Google Colab or
    Jupyter Notebook.
2.  Run the installation cell to install `pandas` and `convokit`.
3.  Run the dataset-loading cells to download GoEmotions and the
    ConvoKit corpus.
4.  Run the cleaning cells to generate processed CSV files and a
    data-cleaning report.
5.  Run the model-dependency installation cell for `transformers`,
    `datasets`, `accelerate`, and `scikit-learn`.
6.  Run the preparation and tokenization cells.
7.  Run the DistilBERT training cell.
8.  Run the test-evaluation and prediction-saving cell.
9.  Review the printed evaluation metrics and the generated prediction
    file.

Run the notebook cells in order. If a runtime is restarted, rerun the
required setup, loading, and preprocessing cells before training.

## Model and Classification Approach

The notebook uses `distilbert-base-uncased` from Hugging Face
Transformers as the base model. It is configured for **multi-label
classification**, because a text can have more than one emotion label.

The notebook: - Converts labels into multi-hot vectors. - Tokenizes text
with a maximum sequence length of 128. - Fine-tunes the model using the
Hugging Face `Trainer`. - Applies sigmoid to model logits to obtain
per-emotion probabilities. - Uses a threshold of 0.5 for the reported
binary predictions in the metric function. - Calculates macro-F1,
micro-F1, and accuracy.

The notebook contains training configurations with different epoch
counts. Use one intended training configuration consistently for the
final experiment, and record the selected settings when reporting
results.

## Generated Files

The notebook is configured to save outputs under:

`/content/emotion_community_project/`

Expected files include:

``` text
emotion_community_project/
├── data/
│   ├── raw/
│   └── processed/
│       ├── goemotions_train.csv
│       ├── goemotions_validation.csv
│       ├── goemotions_test.csv
│       ├── convokit_conversations.csv
│       └── goemotions_test_predictions.csv
└── results/
    └── data_cleaning_report.json
```

The exact files available depend on which notebook cells finish
successfully.

### Prediction file

`goemotions_test_predictions.csv` is intended to contain the test
comment ID, text, emotion labels, and one probability column per
emotion, using names such as `prob_joy` and `prob_sadness`.

### Cleaning report

`data_cleaning_report.json` records data-cleaning information generated
by the notebook.

## Evaluation

The notebook reports: - **Macro-F1:** gives each emotion category equal
weight. - **Micro-F1:** aggregates label-level decisions across
categories. - **Accuracy:** computed by the notebook's current metric
function.

No performance score is stated in this README because the final training
output must be obtained by running the notebook. Report only the metrics
actually produced by the completed run.

## Planned Next Stages

These stages are part of the broader project plan but are not
implemented by the current notebook cells described above:

1.  Apply the trained classifier to messages in the ConvoKit
    conversation corpus.
2.  Construct a user-interaction graph from observed reply
    relationships.
3.  Run an overlapping community detection method such as SLPA or
    BigCLAM.
4.  Compare the overlapping method with a non-overlapping baseline such
    as Louvain.
5.  Aggregate emotion predictions at user and community level.
6.  Add temporal analysis only if reliable timestamps are available.
7.  Optionally build an API and dashboard for exploring graph structure,
    emotion distributions, and results.

## Reproducibility and Limitations

-   Run the notebook from top to bottom and record the runtime, model
    settings, and package versions for the final report.
-   Dataset downloads and pretrained model downloads require internet
    access.
-   Model training can take a substantial amount of time, especially
    without a GPU.
-   The current notebook's cleaning and training cells should be checked
    for consistency before the final experiment, including label-column
    ordering, label encoding, and the selected training configuration.
-   Predicted emotions are model estimates, not definitive statements
    about a person's emotional state.
-   Do not claim community-detection results until the graph and
    community stages have been implemented and evaluated.

## References and Resources

-   GoEmotions dataset:
    https://github.com/google-research/google-research/tree/master/goemotions
-   Hugging Face Transformers: https://huggingface.co/docs/transformers
-   ConvoKit documentation: https://convokit.cornell.edu/documentation/
-   scikit-learn metrics:
    https://scikit-learn.org/stable/modules/model_evaluation.html

## Project Status

**Current notebook scope:** Dataset loading, preprocessing, and
multi-label emotion-classifier training/evaluation.

**Not yet confirmed as completed:** Graph construction, overlapping
community detection, temporal analysis, and web application.
