#!/usr/bin/env python
# coding: utf-8

# # Artificial Intelligence Spring 99 <img src = 'https://ece.ut.ac.ir/cict-theme/images/footer-logo.png' alt="Tehran-University-Logo" width="150" height="150" align="right">
# ## Project Final : Price Estimator Using Regression
# ### Dr. Hakimeh Fadaei
# ### By Omid Vaheb

# ## Introduction:
# In this project, after inspecting data, I prepared and normalized it for implementing learning algorithms on it. The most significant barrier in the way was doing preprocessing for texts in persian but I handled it with Hazm library. The final step was to enhance and set hyperparameters for some regression models to get the maximum accuracy and minimum MSE from each model.
# ## Question:
# In order to predict price of a cellphone in Iran's market we need to build a model using machine learning algorithms. The dataset we used in this project was Divar.com 's real data from 5 years ago. Divar is a site in which peaple post an ad for their product to sell them. First we train the model using the given dataset. This data set consists of Brand, Title of ad,  price, description of ad, date in which the was added, number of images uploaded with the ad.

# Before anything we import libraries needed in the project.

# In[60]:


from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import itertools
import gc
#!pip install hazm
import hazm
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
#!pip install lightgbm
import random
import lightgbm as lgb
from lightgbm import LGBMRegressor
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# Now it is time to inspect data by some simple commands.

# In[2]:


data = pd.read_csv('mobile_phone_dataset.csv')
data.head(10)


# It is clear that we don't have any nan value but there are -1 in prices and there is an index column which does not help.

# In[3]:


data.info()


# In[4]:


data.describe()


# We can see that there are less than 400 rows with price less than 10000 (which is clearly a mistype) so we can drop them because they are below 1% of dataset and are not worth fixing.

# In[5]:


data = data.drop(columns = ['Unnamed: 0'])
data.loc[(data.price <= 10000) & (data.price != -1)]


# In[6]:


dataframe = data.loc[(data.price > 10000) | (data.price == -1)]


# In[7]:


dataframe.loc[(dataframe.price <= 10000) & (dataframe.price != -1)]


# Now we check the upper limit of price and there are not many anomalies so we don't do anything with them.

# In[8]:


dataframe.loc[(dataframe.price > 2000000)]


# In[9]:


dataframe.info()


# In[10]:


dataframe.isnull().sum()


# Now is the time to visualize and check the distribution of data regarding different features. First we check brand and we can see that Samsung is the most common brand in ads and after that is Apple.

# In[11]:


dataframe.brand.describe()


# In[12]:


ax = dataframe['brand'].value_counts().plot(kind = 'bar',
                                    figsize = (15, 10),
                                    title = "Number of Each Brand");
ax.set_xlabel("Brands")
ax.set_ylabel("Count")


# We do this visualization for city of the person who posted ad too.

# In[13]:


dataframe.city.describe()


# In[14]:


ax = dataframe['city'].value_counts().plot(kind = 'bar',
                                    figsize = (15, 10),
                                    title = "Number of Each Brand");
ax.set_xlabel("City")
ax.set_ylabel("Count")


# Now it is time to handle rows with -1 as price. I chose to replace them with mean of their category because it doesn't affect the data that much and droping them would have destroyed randomness of data.

# In[15]:


dataframe.loc[dataframe.price == -1]


# In[16]:


dataframe.tail(10)


# In[17]:


dataframe = dataframe.replace(-1, np.nan)
dataframe['price'] = dataframe.groupby('brand')['price'].transform(lambda grp: grp.fillna(np.mean(grp)))


# In[18]:


dataframe.loc[dataframe.price == -1]


# In[19]:


dataframe.tail(10)


# In the next part I ploted box plot of price to check distribution of this feature.

# In[20]:


sns.boxplot(y = 'price', data = dataframe, showfliers = False)
plt.show()


# In the next part I drew scatterplot of dataset regarding brand, city and image_count which is clear that image count does not give any useful information and we should drop it since the regression line for it is absoloutly a horizontal line.

# In[21]:


plt.figure(figsize=(15,10))
chart = sns.regplot(x = 'brand', y = 'price', data = dataframe,
            scatter_kws = {'alpha' : 0.3}, line_kws = {'color' : 'orange'})


# In[22]:


plt.figure(figsize=(15,10))
chart = sns.regplot(x = 'city', y = 'price', data = dataframe,
            scatter_kws = {'alpha' : 0.3}, line_kws = {'color' : 'orange'})


# In[23]:


plt.figure(figsize=(15,10))
chart = sns.regplot(x = 'image_count', y = 'price', data = dataframe,
            scatter_kws = {'alpha' : 0.3}, line_kws = {'color' : 'orange'})


# In[24]:


dataframe = dataframe.drop(columns = ['image_count'])


# The next step is to handle the date that ad was added to site. At first i break this feature into 2 features of hour and day of week. Now I drew histogram of price day-wise and it is clear that general shape of dataset is similar for all days and this feature does not give us additional information. We can also use information gain for hour of the day and we can see that its gain is less than other featues. So I decided to drop this column too.

# In[25]:


def dateHandler(row): 
    row['AMorPM'] = row['created_at'][-2:]
    row['hour'] = int(row['created_at'][-4:-2])
    if row['AMorPM'] == "PM":
        row['hour'] += 12
    row['day'] = re.split('\W', row['created_at'])[0]
    return row
dataframe = dataframe.apply(dateHandler, axis = 'columns')
dataframe.head(10)


# In[26]:


dataframe = dataframe.drop(columns = ['AMorPM', 'created_at'])
x1 = dataframe.loc[dataframe.day == 'Monday', 'price']
x2 = dataframe.loc[dataframe.day == 'Tuesday', 'price']
x3 = dataframe.loc[dataframe.day == 'Wednesday', 'price']
x4 = dataframe.loc[dataframe.day == 'Thursday', 'price']
x5 = dataframe.loc[dataframe.day == 'Friday', 'price']
x6 = dataframe.loc[dataframe.day == 'Saturday', 'price']
x7 = dataframe.loc[dataframe.day == 'Sunday', 'price']
plt.figure(figsize = (15,10))
kwargs = dict(alpha = 0.5, bins = 100)
plt.hist(x1, **kwargs, color = 'b', label = 'Monday')
plt.gca().set(title = 'Frequency Histogram of Price in Moondays', ylabel = 'Frequency', xlabel = 'Price')
plt.figure(figsize = (15,10))
plt.hist(x2, **kwargs, color = 'r', label = 'Tuesday')
plt.gca().set(title = 'Frequency Histogram of Price in Tuesdays', ylabel = 'Frequency', xlabel = 'Price')
plt.figure(figsize = (15,10))
plt.hist(x3, **kwargs, color = 'g', label = 'Wednesday')
plt.gca().set(title = 'Frequency Histogram of Price in Wednesdays', ylabel = 'Frequency', xlabel = 'Price')
plt.figure(figsize = (15,10))
plt.hist(x4, **kwargs, color = 'c', label = 'Thursday')
plt.gca().set(title = 'Frequency Histogram of Price in Thursdays', ylabel = 'Frequency', xlabel = 'Price')
plt.figure(figsize = (15,10))
plt.hist(x5, **kwargs, color = 'm', label = 'Friday')
plt.gca().set(title = 'Frequency Histogram of Price in Fridays', ylabel = 'Frequency', xlabel = 'Price')
plt.figure(figsize = (15,10))
plt.hist(x6, **kwargs, color = 'y', label = 'Saturday')
plt.gca().set(title = 'Frequency Histogram of Price in Saturdays', ylabel = 'Frequency', xlabel = 'Price')
plt.figure(figsize = (15,10))
plt.hist(x7, **kwargs, color = 'k', label = 'Sunday')
plt.gca().set(title = 'Frequency Histogram of Price in Sundays', ylabel = 'Frequency', xlabel = 'Price')


# In[27]:


dataframe = dataframe.drop(columns = ['hour', 'day'])
dataframe.head(10)


# Now we vectorize brand and city using one hot encoding that we have used before. In this method you put a column for each possible value for feature and the value of that column is 1 if the feature said before is equal to column's value. In this dataset brand has 9 options and city has 9 too so one hot encoding is a suitable option.

# In[28]:


dataframe = pd.concat([dataframe, pd.get_dummies(dataframe['city'], prefix = 'city')], axis = 1)
dataframe = pd.concat([dataframe, pd.get_dummies(dataframe['brand'], prefix = 'brand')], axis = 1)
dataframe = dataframe.drop(columns = ['brand', 'city'])


# In[29]:


dataframe.head(10)


# The final step of preprocessing is to process 2 remaining unchanged columns: Title and description
# These columns contain some persian texts and they also have some few english words and characters so by using hazm and nltk libraries i cleaned these exts. The actions taken were normalization, tokenizing and lemmatizing(we saw in previous projects that it is more through and accurate than stemming). I also created my own set of stopwords regarding dataset.

# In[30]:


def preprocessTextofColumn(columnName):
    normalizer = hazm.Normalizer()
    tokenizer = hazm.WordTokenizer()
    lemmatizer = hazm.Lemmatizer()
    ENGStopWords = set(stopwords.words('english'))
    PERStopWords = set(hazm.utils.stopwords_list())
    PERStopWords = PERStopWords.union({':','ی','ای',',','،','(',')',':',';','-','_','.','/','+','=','?'})
    stopWords = ENGStopWords.union(PERStopWords)
    allWords = []
    for index, row in dataframe.iterrows():
        text = row[columnName]
        normalizedText = normalizer.affix_spacing(text)
        words = tokenizer.tokenize(normalizedText)
        filteredWords = []
        for word in words:
            if not word in stopWords:
                filteredWords.append(lemmatizer.lemmatize(word.lower()))
        allWords.append(filteredWords)
    newColumnName = columnName + 'Words'
    dataframe[newColumnName] = allWords
    return dataframe


# In[31]:


dataframe = preprocessTextofColumn('desc')
dataframe = preprocessTextofColumn('title')


# In[32]:


dataframe = dataframe.drop(columns = ['desc', 'title'])
dataframe.head(10)


# Now we split the dataset into train and test using sklearn's command train_test_split. It is splited into 80% train and 20% test.

# In[33]:


train = dataframe.drop(columns = ['price'])
test = dataframe['price']
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 7)
print('Train size: {}, Test size: {}' .format(X_train.shape, X_test.shape))


# Te final thing before learning is to convert these list of words , that came from title and description, into features. I used one hot encoding here but there would be over 15000 features if I had used all words so we count number of occurance for each word and keep only 250 words which were more frequent.

# In[34]:


titleCounts = {}
descCounts = {}
titleDictionary = {}
descDictionary = {}
for index, row in X_train.iterrows():
    for word in row['titleWords']:
        if word not in titleCounts:
            titleCounts[word] = 1
        else:
            titleCounts[word] += 1
    for word in row['descWords']:
        if word not in descCounts:
            descCounts[word] = 1
        else:
            descCounts[word] += 1 
titleDictionary = {k: v for k, v in sorted(titleCounts.items(), key = lambda item: item[1], reverse = True)}
descDictionary = {k: v for k, v in sorted(descCounts.items(), key = lambda item: item[1], reverse = True)}


# In[35]:


print(len(titleDictionary))
print(titleDictionary)


# In[36]:


print(len(descDictionary))
print(descDictionary)


# I also removed two frequent words from features because they were general for all mobiles so they would not helped us to get a more accurate price.

# In[37]:


del titleDictionary['گوش']
del descDictionary['گوش']
del titleDictionary['موبایل']
del descDictionary['موبایل']
titleDictionary = dict(itertools.islice(titleDictionary.items(), 250))
descDictionary = dict(itertools.islice(descDictionary.items(), 250))


# In[38]:


for key in titleDictionary:
    X_train['title_' + key] = 0
    X_train['title_' + key] = X_train['titleWords'].apply(lambda x : 1 if key in x else 0)
    X_test['title_' + key] = 0
    X_test['title_' + key] = X_test['titleWords'].apply(lambda x : 1 if key in x else 0)
for key in descDictionary:
    X_train['desc_' + key] = 0
    X_train['desc_' + key] = X_train['descWords'].apply(lambda x : 1 if key in x else 0)
    X_test['desc_' + key] = 0
    X_test['desc_' + key] = X_test['descWords'].apply(lambda x : 1 if key in x else 0)
X_train = X_train.drop(columns = ['descWords', 'titleWords'])
X_test = X_test.drop(columns = ['descWords', 'titleWords'])


# In[39]:


X_train.head(10)


# In[40]:


X_test.head(10)


# Now is the the time to learn the dataset using some regression-based algorithms. Our measure of accuracy for models is minimum square error because it shows us that in general how far was our price prediction.

# First one is a simple linear regressor. LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. This model works realy good and its MSE is about 290000 tomans meaning that in average we predicted the price of mobiles by an error of 290000 tomans.

# In[41]:


model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)
print('MSE for Linear Regression:', np.sqrt(mean_squared_error(y_test, predictions)))


# Now we try Ridge resgression. Traditional linear fitting involves minimizing the RSS (residual sum of squares). In ridge regression, a new parameter is added, and now the parameters will minimize:
# ![image.png](attachment:image.png)
# 
# Where lambda is a tuning parameter. This parameter is found using cross-validation as it must minimize the test error. Therefore, a range of lambdas is used to fit the model and the lambda that minimizes the test error is the optimal value.
# Here, ridge regression will include all p predictors in the model. Hence, it is a good method to improve the fit of the model, but it will not perform variable selection. In order to tune its hyperparameter, alpha, we try some common valus for it and draw MSE of train and test for each alpha and the best error that i achieved using idge is 294000 whixh is for alpha = 0.1 .

# In[62]:


alpha = [0.1, 0.5, 1, 2, 4, 6] 
testAccuracies = []
trainAccuracies = []
for i in alpha:
    model = Ridge(solver = "sag", random_state = 7, alpha = i, copy_X = True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    testAccuracies.append(np.sqrt(mean_squared_error(y_test, predictions)))
    predictions = model.predict(X_train)
    trainAccuracies.append(np.sqrt(mean_squared_error(y_train, predictions)))
fig, ax = plt.subplots()
ax.plot(alpha, testAccuracies)
ax.scatter(alpha, testAccuracies, label = 'Test')
ax.plot(alpha, trainAccuracies)
ax.scatter(alpha, trainAccuracies, label = 'Train')
plt.legend(loc = 'upper left');
plt.title("Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.show()


# In[63]:


print('MSE for Linear Regression:', testAccuracies[0])


# One other regresser I used is SVM. SVM or Support Vector regression is a type of Support vector machine that supports linear and non-linear regression. As it seems in the below graph, the mission is to fit as many instances as possible between the lines while limiting the margin violations. The violation concept in this example represents as ε (epsilon). SVR requires the training data:{ X, Y} which covers the domain of interest and is accompanied by solutions on that domain. The work of the SVM is to approximate the function we used to generate the training set to reinforce some of the information we’ve already discussed in a classification problem.

# In[44]:


Cs = [0.1, 0.5, 0.8, 1, 2] 
testAccuracies = []
trainAccuracies = []
for c in Cs:
    model = SVR(C = c, max_iter = 200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    testAccuracies.append(np.sqrt(mean_squared_error(y_test, predictions)))
    predictions = model.predict(X_train)
    trainAccuracies.append(np.sqrt(mean_squared_error(y_train, predictions)))
fig, ax = plt.subplots()
ax.plot(Cs, testAccuracies)
ax.scatter(Cs, testAccuracies, label = 'Test')
ax.plot(Cs, trainAccuracies)
ax.scatter(Cs, trainAccuracies, label = 'Train')
plt.legend(loc = 'upper left');
plt.title("Error for each C")
plt.xlabel("C")
plt.ylabel("Error")
plt.show()


# We can see that our error doubled which shows us that SVM is not a good nmodel for this dataset.
# The last model I used is LGBM which is a gradient boosting model that uses tree-based learning algorithms. Unfortunately because of limited time i couldn't completely implement this model because it has many hyperparameters and it had some barriesr for persian text learning. But from my implementation, it is seen that it won;t do any better that other models since its error is nan in most cases.

# In[61]:


lgb_model = LGBMRegressor(subsample=0.9)
params = {'learning_rate':uniform(0, 1),
          'n_estimators': sp_randint(200, 1500),
          'num_leaves': sp_randint(20, 200),
          'max_depth': sp_randint(2, 15),
          'min_child_weight': uniform(0, 2),
          'colsample_bytree': uniform(0, 1),
         }
lgb_random = RandomizedSearchCV(lgb_model, param_distributions=params, n_iter=10, cv=3, random_state=7, 
                                scoring='neg_root_mean_squared_error', verbose=10, return_train_score=True)
lgb_random = lgb_random.fit(X_train, y_train)


# ## Conclusion:
# We got an MSE of around 290000 for out price predictor which is a good error because of 5 reasons:
# 1 - Dataset was from Divar.com which is a site that peaple put their own ads in ordet to sell their products so the prices are not accurate enough and it has many noises such as misclicks and so on. Dataset of a site like Torob would be much better because real time sellers put prices there and they are more accurate.
# 2 - Our dataset did not have many usefull features and we had to drop most of them.
# 3 - Description of a mobile ad is not a very useful thin in order to predict its price because ther could have been many errors and noises in it and a table of mobile features could have been a perfetct dataset for this purpose.
# 4 - Hazm tool was not as good as nltk and needs a lot more develpoment.If this dataset was in english, the results would have been mich higher.
# 5 - Data might have not been enough for this task because there are thousands of mobile models and a 60000 dataset won't be enough for this purpose.
# At the end I think we could have got a lower error if:
# 1 - we used tf-idf measure instead of countong words
# 2 - Neural networks such as RNN and CNN would have been better for this purpose
# 3 - Workin more on preprocess of persian texts could have been a realy effective job because there were many bugs in Hazm.

# In[ ]:





# In[ ]:




