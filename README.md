# Disaster Tweets Classification using RNN

This is part of a course assignment in our Deep Learning class.
NLP project for Kaggle competition: Classify disaster tweets using RNN.

## Overview

**Goal:** Build an ML model to predict which tweets are about real disasters vs. misclassified tweets.

**Competition:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

**Evaluation Metric:** F1 Score

## Results

Simple RNN: 0.5700
LSTM: 0.0000 (failed) 
GRU: 0.0000 (failed)
Bidirectional LSTM: **0.7367** 

Note: Both LSTM and GRU likely got stuck at the local minima and failed to produce F1s.

**Best Model:** Bidirectional LSTM achieved F1 = 0.74 on val.

## Architecture

**Approach:** 
- Learned word embeddings and RNN

**Models:**
- Simple RNN (baseline)
- LSTM 
- GRU
- Bidirectional LSTM (best)

**Key Components:**
- Embedding layer (100-dim)
- RNN layer (64 units)
- Dropout (0.5)
- Dense output (sigmoid)

## Files

- `disaster_tweets.ipynb` - Notebook with all steps taken
- `submission.csv` - Kaggle competition submission file
- `requirements.txt` - Dependencies

Note: To run the full notebook you will need the sample submission file from Kaggle.

## Setup
```
# Clone repository
git clone https://github.com/YOUR_USERNAME/disaster-tweets-classification.git

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
# https://www.kaggle.com/c/nlp-getting-started/data
```

## Project Outline

1. **Problem Statement** - Task definition and approach
2. **EDA** - Data exploration and visualization
3. **Model Architecture** - RNN implementations
4. **Results & Analysis** - Training, evaluation, F1 scores
5. **Conclusion** - Findings and future improvements

## Key Learnings

- Simple RNN demonstrates vanishing gradient problem (as discussed during class)
- Gated architectures such as LSTM are essential for sequence modeling
- Bidirectional processing improves context understanding and let to the best model performance
- Random initialization significantly impacts convergence

## References

- Deep Learning textbook (Goodfellow, Bengio, Courville) - Chapter 10 (RNNs)
- Kaggle competition: https://www.kaggle.com/c/nlp-getting-started

---

**Author:** Philipp Adrian Pohlmann
**Course:** Deep Learning Intro Course (Boulder - MS CS)  
**Date:** November 2025
