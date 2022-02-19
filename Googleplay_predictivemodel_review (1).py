#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams as rcp
import math
plt.rc("font", size=13)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dfPlayStore = pd.read_csv("googleplaystore_v6.csv")


# In[4]:


dfPlayStore.rename(columns={'Content Rating': 'Content_Rating','Last Updated':'Last_Updated','Current Ver':'Current_Version','Android Ver':'Android_Version'}, inplace=True)


# In[5]:


dfPlayStore.head(10)


# In[6]:


dfPlayStore.tail(10)


# In[7]:


dfPlayStore.shape


# In[8]:


dfPlayStore.describe()


# In[9]:


dfPlayStore.isnull().sum()


# In[10]:


dfPlayStore.corr()


# In[11]:


total = dfPlayStore.isnull().sum().sort_values(ascending=False)
percent = (dfPlayStore.isnull().sum()/dfPlayStore.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)


# In[12]:


dfPlayStore['Rating'].describe()


# In[13]:


# rating distibution 
rcp['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(dfPlayStore.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)


# In[14]:


Category_App_df = dfPlayStore.groupby('Category')["App"].agg(["count"]).reset_index() 
Category_App_df


# In[15]:


Category_Rating_df = dfPlayStore.groupby('Category')["Rating"].agg(["mean"]).reset_index() 
Category_Rating_df


# In[16]:


Category_Installs_df = dfPlayStore.groupby('Category')["Installs"].agg(["mean"]).reset_index().sort_values('mean',ascending=False)
Category_Installs_df


# In[17]:


Category_Installs_df = dfPlayStore.groupby('Category')["Installs"].agg(["mean"]).reset_index().sort_values('mean',ascending=False)
#Category_Installs_df

plt.rcParams['figure.figsize'] = (100,6)
plt.rcParams['xtick.labelsize'] = (15.0)
plt.rcParams['ytick.labelsize'] = (15.0)

Category_Installs_df.plot(kind='bar',    # Plot a bar chart
        legend=False,    # Turn the Legend off
        width=0.75,      # Set bar width as 75% of space available
        figsize=(16,11),  # Set size of plot in inches
                          #subplots=True,
        color=[plt.cm.Paired(np.arange(len(Category_Installs_df)))])
plt.xlabel('Category')
plt.ylabel('Number Of Installs')


# In[18]:


plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'purple',data=dfPlayStore[dfPlayStore['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)


# In[19]:


g = sns.catplot(x = "Category",y = "Rating",data = dfPlayStore, kind = "violin", height = 10, palette = "Set1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set( xticks=range(0,34))
g = g.set_ylabels("Rating")
plt.title('Violin Plot Of Rating VS Category',size = 20)


# In[20]:


sns.set(style="ticks")
g = sns.catplot(x = "Category",y = "Rating",data = dfPlayStore, height = 10, palette = "Set1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set( xticks=range(0,34))
g = g.set_ylabels("Rating")
plt.title('Tick Plot of Installs VS Category',size = 20)


# In[21]:


# Data to plot
labels =dfPlayStore['Type'].value_counts(sort = True).index
sizes = dfPlayStore['Type'].value_counts(sort = True)


colors = ["blue","yellow"]
explode = (0.1,0)  # explode 1st slice
 
rcp['figure.figsize'] = 8,8
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percentage of Free V/S Paid Apps in the App Store',size = 20)
plt.show()


# In[22]:


Category_Paid_Price_df = dfPlayStore[dfPlayStore['Price_Dollars'] > 0]
Category_Price_df = Category_Paid_Price_df.groupby('Category')["Price_Dollars"].agg(["mean","count"]).reset_index().sort_values('mean',ascending=False)
Category_Price_df


# In[23]:


Category_Installs_df = dfPlayStore.groupby('Category')["Installs"].agg(["mean"])
Category_Rating_df = dfPlayStore.groupby('Category')["Rating"].agg(["mean"])


Customer_Installs = Category_Installs_df['mean']
Customer_App_Rating = Category_Rating_df['mean']
plt.scatter(Customer_Installs, Customer_App_Rating, color = 'blue', edgecolors='red')
plt.xlabel('Number of installs per app')
plt.ylabel('Application rating per app')
plt.title('Does Application Rating Affect Number of Installs?')
plt.show()


# In[24]:


Category_Paid_Price_df = dfPlayStore[dfPlayStore['Installs'] < 18000]
plt.scatter(Category_Paid_Price_df['Installs'], Category_Paid_Price_df['Rating'], color ='green', edgecolors = 'red')
plt.xlabel('Number of installs per app')
plt.ylabel('Application rating per app')
plt.title('Does Application Rating Affect Number of Installs?')
plt.show()


# In[25]:


free_apps = dfPlayStore[(dfPlayStore['Type'] == 'Free')]
free_apps.reset_index(inplace= True)

paid_apps = dfPlayStore[(dfPlayStore['Type'] == 'Paid')]
paid_apps.reset_index(inplace= True)


# In[26]:


from scipy import stats
stats.levene(free_apps['Installs'], paid_apps['Installs'])


# In[27]:


stats.ttest_ind(free_apps['Installs'], paid_apps['Installs'])


# In[28]:


stats.ttest_ind(free_apps['Installs'],paid_apps['Installs'], equal_var=False)


# In[29]:


Ratings_DataX = dfPlayStore[['Price_Dollars','Type','Reviews','Size_MB','Content_Rating']]


# In[30]:


Ratings_DataX.head(10)


# In[31]:


Ratings_DataX = pd.get_dummies(Ratings_DataX)
Ratings_DataX.head(10)


# In[32]:


Ratings_DataY = dfPlayStore[['Rating']]


# In[33]:


Ratings_DataY.head(10)


# In[34]:


Installs_DataX = dfPlayStore[['Price_Dollars','Type','Reviews','Size_MB']]
Installs_DataX = pd.get_dummies(Installs_DataX)


# In[35]:


Installs_DataX.head(10)


# In[36]:


Installs_DataY = dfPlayStore[['Installs']]


# In[37]:


Installs_DataY.head(10)


# In[38]:


import statsmodels.formula.api as smf
from scipy import stats


# In[39]:


rating_model = smf.ols("Rating ~ Price_Dollars + Type + Reviews + Size_MB + Content_Rating", data= dfPlayStore).fit()

rating_model.summary()


# In[40]:


stats.probplot(rating_model.resid, plot= plt)
plt.title("Model 1 Residuals Probability Plot")


# In[41]:


installs_model = smf.ols("Installs ~ Price_Dollars + Type + Reviews + Size_MB", data= dfPlayStore).fit()

installs_model.summary()


# In[42]:


stats.probplot(installs_model.resid, plot= plt)
plt.title("Model 2 Residuals Probability Plot")


# In[43]:


from sklearn.model_selection import train_test_split  
RX_train, RX_test, Ry_train, Ry_test = train_test_split(Ratings_DataX, Ratings_DataY, test_size=0.3, random_state=0)  


# In[44]:


from sklearn.linear_model import LinearRegression  
Ratings_regressor = LinearRegression()  
Ratings_regressor.fit(RX_train, Ry_train)


# In[45]:


print(Ratings_regressor.intercept_)


# In[46]:


print(Ratings_regressor.coef_)


# In[47]:


Ry_pred = Ratings_regressor.predict(RX_test)


# In[49]:


from sklearn.metrics import mean_squared_error, r2_score

print('Coefficients: \n', Ratings_regressor.coef_)

print("Mean squared error: %.2f"
      % mean_squared_error(Ry_test, Ry_pred))

print('Variance score: %.2f' % r2_score(Ry_test, Ry_pred))


# In[57]:


Installs_XData = dfPlayStore[['Price_Dollars','Reviews','Size_MB','Type_App']]
Installs_YData = dfPlayStore[['Installs_Greater']]


# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  
from sklearn import metrics

InstallsX_train, InstallsX_test, InstallsY_train, InstallsY_test = train_test_split(Installs_XData, Installs_YData, test_size=0.4, random_state=0)


# In[59]:


import statsmodels.api as sm


# In[60]:


Installs_logit_model = sm.Logit(InstallsY_train,InstallsX_train)


# In[61]:


LogitResult=Installs_logit_model.fit()
print(LogitResult.summary())


# In[62]:


print('Price_Dollars: {:.2f}'.format(np.exp(0.0011)))
print('Reviews: {:.2f}'.format(np.exp(1.005e-05)))
print('Size_MB: {:.2f}'.format(np.exp(0.0243)))


# In[63]:


Installs_logisticreg = LogisticRegression(C=1e9)
Installs_logisticreg.fit(InstallsX_train, InstallsY_train)
InstallsY_pred = Installs_logisticreg.predict(InstallsX_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(Installs_logisticreg.score(InstallsX_test, InstallsY_test)))


# In[64]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

confusion_matrix = confusion_matrix(InstallsY_test, InstallsY_pred)
classification_report = classification_report(InstallsY_test, InstallsY_pred)
accuracy_score = accuracy_score(InstallsY_test, InstallsY_pred)

print(confusion_matrix)
print(classification_report)
print(accuracy_score)


# In[68]:


#Conclusion:
#Correct predictions = 112 + 2431 = 2533

#Errors = 0 + 1203 = 1203

#Hence, we conclude that 67% of the times our model accurately predicts that a given app will be installed 100,000 plus times, based on the factors: Price_Dollars, Reviews, Size_MB, and Type_App.


# In[ ]:




