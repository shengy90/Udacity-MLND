# Machine Learning Engineer Nanodegree
## Capstone Proposal
Sheng Chai 
30th March, 2017

## Proposal

### Domain Background

This Capstone Project is inspired by the competition Santander hosted on Kaggle back in 2016. The aim of the project is to help Santander identify which customers are more likely to be unhappy so that the bank can take proactive steps to reduce customer churn rates by getting in touch with unhappy customers to improve the relationship so they won't leave the bank in the future.

Background on churn (attrition) rates:
Churn rate refers to the dropout rate of a customers, i.e. what % of customers that stopped/ ended their relationship with a company. Company's success largely depends on the size of their customer base - the bigger their customer base is, the higher their revenue is going to be. In order for a customer to successfully grow their customer base, the customer growth rate (rate of acquiring new customer) must be higher than their customer churn rate (rate of existing customers ending their relationship).

Santander has sought the Kaggle's community help to identify customers that are more likely to leave the company so that the Bank can take active steps to improve the relationship before it's too late to improve customer loyalty and thus reducing churn rates.

In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience.

In this section, provide brief details on the background information of the domain from which the project is proposed. Historica
### Problem Statement

The aim of this project is to develop an algorithm to detect unhappy customers. This is a classification problem with 2 outputs: 0 = happy and 1 = unhappy. The dataset given comprises of hundreds of anonymised features and these features will be used to classify happy vs unhappy customers.

Because this is a Kaggle dataset, the testing dataset will not have a 'TARGET' value. To get testing error, solution will have to be uploaded to Kaggle to get the accuracy.

This sounds like quite a straightforward and simple binary classification problem.


### Datasets and Inputs

The dataset comprises of 370 features and the features are 'anonymised'. The variables are numeric - some will be discrete (categorical) and some will be continuous. Because the features are 'anonymised', there is no way to tell what each variables actually mean. However that shouldn't affect our ability to build a predictive model. It doesn't really matter whether the column 'var3' means country or products or occupation - if it's a good predictor, then it will be regardless of what it actually means. The dataset is structured / tabulated which makes it quite ideal for decision tree classification.


All data are prepared and provided by the courtesy of Santander bank and due to confidentiality purpose, no more information can be given beyond what's already given.

As a result, the first step before we can do anything is to look through all this 370 features and understand more on their characteristics, for example if there are any duplicates, or if it's possible to determine if they are categorical etc. After understanding the characteristics of the data, then we can proceed to cleansing the data if needed, engineer additional features, or even apply techniques such as PCA etc to reduce the dimensionality of the data and shrink the 370 features into something more manageable.

### Solution Statement

Before we can start modelling, we need to reduce the dimensionality of the data; 370 features is quite excessive. This can be done by removing highly correlated variables, duplicate variables, or using other dimension-reduction techniques such as PCA etc. After the data is cleansed and new condensed features are engineered, then we can proceed with modelling.

There are many approaches we can go about doing this:
 - Decision Tree Classification techniques such as random forest/ boosted trees etc. Decision Trees are great for structured and tabulated data. However this method is prone to over-fitting, but can be addressed using ensemble methods. In fact, random forest/ boosted trees are all ensemble methods that deals with over-fitting.
 - Logistic Regression are great for binary classification task such as this.
 - Neural Networks are great for finding hidden patterns within data for classification task.
 - Stacked models - combination of the above/ additional methods 

The model can then be measured by many metrics used to measure classifiers, such as the F1-score (or the confusion matrix), or the area under the ROC curve (Receiver operating characteristic, which is a plot of true positive vs false positive rates).


### Benchmark Model
_(approximately 1-2 paragraphs)_

As this competition was held in 2016 on data (and has now concluded), there are many existing solutions that we can benchmark our prediction against. Currently,the best score is held by 'Shize & Nir' with an AUC score of 0.829072.

The team that won the first place didn't upload their solutions on to Kaggle. However, I've found the solution for the third-prize on Kaggle and their model could be use as a suitable benchmark.

https://github.com/diefimov/santander_2016/blob/master/README.pdf

As this is a Kaggle competition, we would be solving the same problem and would be using the same evaluation metric (which will be covered in the next paragraph) to compare our model against each other. In this particular case, the metric specified by Santander is the area under the ROC curve.


### Evaluation Metrics


Santander has specified that the evaluation metric we should be using is the area under the ROC (Receiver operating characteristic) curve. The ROC curve is a graphical plot for binary classifier and shows the relationship between the 'True postivie rate' and the 'false positive rate'. 

Apart from the area under the ROC curve, there are other metrics that can be used, for example the F1 score, area under the PR curve etc. However these other metrics are not relavent to this project due to the nature of this Kaggle competition.


### Project Design


##### Part 1: Exploring the data
The dataset consists of 370 variables, which may be categorical or discrete. Because the dataset is anonymised, we have no information on what each variable means. Therefore, we cannot decide whether or not we should include/ exclude a column just by looking at the column name. Instead, we might have to look at each individual variable, then look at the relationship between each variable vs the target to understand what's in each column and if there are any correlation between the variables and the target.

However, because of the large number of variables, manual investigation may be very inefficient. Instead of investigating each variable manually, we could look to some automated feature selection strategy such using random forest or other methods to determine the 'feature importance' which can help us to prune the amount of features.

##### Part 2: Feature Engineering
After getting rid of redundant features, we can then engineer new features if necssary to create more predictive features. Depending on the amount of features, we can also use some dimensionality reduction methods such as PCA to engineer new features and reduce the amount of redundant features.

##### Part 3: Train the model
We can start creating the model after we have obtain a list of useful features that we can use to create our model. Because this is a binary classification task, we can use many different classifiers to create our binary classifier. Example of classifiers are:

 - Logistic Classifier : logistic classifer is great for creating binary classifiers, especially if we're trying to solve a linear problem. However, depending on the features we've decided to use, there might be some non-linearity in our features and logistic regression might not be able to cope so well.
 
 - Neural Network Classifier : neural network is great for finding hidden patterns in our dataset and is great for classification tasks, especially if there is non-linearity in the dataset. However, neural network is computationally expensive, and can be prone to overfitting. However, there are many methods we can deal with overfitness, such as 'dropouts' which is a commonly used technique in deep learning.
 
 - Decision tree classifiers : decision tree classifier is great for classification tasks and can be easily interpreted. However it's very unstable as just a slight purturbation in the dataset can result in a very different-looking tree. However there are many variations of decision tree classifiers that can be used to address the over-fitness e.g.:
     -- Random Forest
     -- Bagged Trees
     -- Boosted Trees

Which classifier to use will depend on the final set of features we want to keep for training the model. If all remaining features are linearly separable then logistic classfier can be a good choice. Otherwise, decision tree is probably better compared to neural network due as this is a simple classification task is training a decision tree is computationally a lot cheaper. Since the task only want a simple 'happy' vs 'not happy' prediction (i.e. we aren't required to determine the 'probabilities'), decision tree is probably the more appropriate choice.

##### Part 4: Validating and Evaluating the model

After the model's been trained, it'll first be tested on an cross-validation dataset (which is a subset of the training set) to determine the optimum hyper-parameter. To reduce overfitness on the training data, we can use k-fold validation method to choose the best hyper-parameters that gives the least cross-validation error.

Once the parameters for the final model has been determined, we will compute the outcomes on the testing set, then upload to Kaggle's server where they will tell us the 'public'and 'private' scores.

Public scores refers to the scores using the 'testing' dataset, whilst 'private' scores refers to the scores using 50% of testing data + 50% unseen data. Difference between public and private scores is to prevent people from overfitting to the public scores. Inclusing another 50% of unseen data ensures that the model is not overfitted.


References:
 1. http://machinelearningmastery.com/an-introduction-to-feature-selection/
 2. https://www.kaggle.com/c/santander-customer-satisfaction
 3. https://github.com/diefimov/santander_2016/blob/master/README.pdf
 4. https://en.wikipedia.org/wiki/Random_forest
 5. https://github.com/ctufts/Cheat_Sheets/wiki/Classification-Model-Pros-and-Cons
 6. http://www.investopedia.com/terms/c/churnrate.asp
 7. https://en.wikipedia.org/wiki/Churn_rate

-----------

