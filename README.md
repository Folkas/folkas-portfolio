# About Portfolio

This repository contains Folkas' portfolio with data science projects, collected during studies at Turing College.

## 1. Project with SQL

**Files**: [Car_pricing_model.ipynb](https://github.com/Folkas/folkas-portfolio/blob/main/Car_pricing_model.ipynb) or [Google colab link](https://colab.research.google.com/drive/1OzMO6M32E2wydzPkVF9MKhETZ5-z7QjY?usp=sharing), [Flask API](https://github.com/Folkas/project_24), [scraping class for en.autoplius.lt](https://github.com/Folkas/autoplius-scraper)

**Technologies used**: ML projects, Linear Regression, SQL (basic), Maintaining ML models, Flask, PostgreSQL, BeautifulSoup

**Description**: This capstone project is dedicated to solve a business problem. Let's say that I'm opening a second-hand car shop. I need to set price for each car, but I don't know its value in the Lithuanian market. Therefore, I'm going to create a model to predict car's price based on several attributes: manufacturing date, engine size (in liters), engine power (in kW), mileage (in km) and whether it has automatic or manual gearbox. My model will use the scraped data from en.autoplius.lt website. The model will help me to evaluate car's value in euros and keep my shop economically sustainable.

## 2. Machine learning capstone project (Adoption prediction)
**Files**: [Adoption prediction notebook](https://github.com/Folkas/folkas-portfolio/blob/main/Adoption_prediction.ipynb) or [Google colab link](https://colab.research.google.com/drive/1S82OdNIArivlAt6CAQSqtsiceJU-u1_n?usp=sharing)

**Technologies used**: EDA (intermediate), Matplotlib, Pandas (advanced), ML projects, feature engineering, Hyperparameter tuning (intermediate), Model selection, Scikit-learn

**Description**: I decided to participate in Kaggle's PetFinder.my Adoption Prediction competition for my 3rd module's capstone project. The task requires to predict pet adoption speed (0-4) based on information about pet (type, age, breed, health, etc.) I've created several linear, tree and gradient boosted tree models for the competition. The highest quadratic weighted kappa score was achieved using Light Gradient Boosting Classifier.

## 3. NLP project (Toxic comment multi-label classifier)
**Files**: [Toxic comment classifier](https://github.com/Folkas/folkas-portfolio/blob/main/Toxic_comment_classifier.ipynb) or [Google colab link](https://colab.research.google.com/drive/1RDnteWVArrusWavl3U2zEie8TpAY7ZW2?usp=sharing)

**Technologies used**: EDA (intermediate), Natural language processing, Attention mechanism, Data cleaning, PyTorch Lightning, BERT models (HuggingFace library)

**Description**: this project was my 2.2 sprint project. This task is a Kaggle's Toxic Comment Classification Challenge, where I had to create a model to identify (0 or 1) toxic comments and classify them into 6 multi-label categories. After performing data cleaning, I used Pytorch Lightning framework to create classes for dataset and DistilBertForSequenceClassification model. During the training, I logged data with Tensorboard for visualizations and then tested its performance using several metrics (classification report, AUROC score, confusion matrix.
