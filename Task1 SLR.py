# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

cc = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Simple Linear Regression/calories_consumed.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

cc.columns = "Weightgained", "CaloriesConsumed" ##renaming so that no sapces is there otherwise error.

cc.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = cc.Weightgained, x = np.arange(1, 15, 1)) #numpy.arange([start, ]stop, [step, ], dtype=None) -> numpy.ndarray
plt.hist(cc.Weightgained) #histogram
plt.boxplot(cc.Weightgained) #boxplot

plt.bar(height = cc.CaloriesConsumed, x = np.arange(1, 15, 1))
plt.hist(cc.CaloriesConsumed) #histogram
plt.boxplot(cc.CaloriesConsumed) #boxplot

# Scatter plot
plt.scatter(x = cc['Weightgained'], y = cc['CaloriesConsumed'], color = 'green') 

# correlation
np.corrcoef(cc.Weightgained, cc.CaloriesConsumed) #0.94699101 - Strong Relation (greater than 0.85)

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(cc.Weightgained, cc.CaloriesConsumed)[0, 1]
cov_output #237669.4505494506

# cc.cov()


# Import library
import statsmodels.formula.api as smf # api gives Ordinary Least Square - sum of squares,(y-y^)^2+..., best fit line

# Simple Linear Regression
model = smf.ols('CaloriesConsumed ~ Weightgained', data = cc).fit()
model.summary() # p value should always less than 0.05 - if yes then it is statistically significant, 
#R-squared:0.897 = it is more than 0.8 => strong correlation => goodness of fit
# Equation is, y = 1577.2007+2.1344(Weightgained) -> B0+B1x

pred1 = model.predict(pd.DataFrame(cc['Weightgained']))

# Regression Line
plt.scatter(cc.Weightgained, cc.CaloriesConsumed)
plt.plot(cc.Weightgained, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = cc.CaloriesConsumed - pred1 # Actual - Predicted
res_sqr1 = res1 * res1 # r square
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 #232.8335007096088

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(cc['Weightgained']), y = cc['CaloriesConsumed'], color = 'brown')
np.corrcoef(np.log(cc.Weightgained), cc.CaloriesConsumed) #correlation = 0.93680369 (greater than 0.85)

model2 = smf.ols('CaloriesConsumed ~ np.log(Weightgained)', data = cc).fit()
model2.summary() #R-squared:0.878 = it is more than 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = -1911.1244+774.1735(Weightgained) -> B0+B1x

pred2 = model2.predict(pd.DataFrame(cc['Weightgained']))

# Regression Line
plt.scatter(np.log(cc.Weightgained), cc.CaloriesConsumed)
plt.plot(np.log(cc.Weightgained), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = cc.CaloriesConsumed - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 #253.5580403936626


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = cc['Weightgained'], y = np.log(cc['CaloriesConsumed']), color = 'orange')
np.corrcoef(cc.Weightgained, np.log(cc.CaloriesConsumed)) #correlation =  0.89872528 - Strong Relation (greater than 0.85)

model3 = smf.ols('np.log(CaloriesConsumed) ~ Weightgained', data = cc).fit()
model3.summary() #R-squared: 0.808= it is more than 0.8 => moderate correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 7.4068+0.0009(Weightgained) -> B0+B1x


pred3 = model3.predict(pd.DataFrame(cc['Weightgained']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(cc.Weightgained, np.log(cc.CaloriesConsumed))
plt.plot(cc.Weightgained, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = cc.CaloriesConsumed - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 #272.4207117048494


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(CaloriesConsumed) ~ Weightgained + I(Weightgained*Weightgained)', data = cc).fit() # y ~ X + I X^2 + I X^3
model4.summary() #R-squared: 0.852 = it is somewhat equal to  0.8 => moderate correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 7.2892 +0.0017(Weightgained) -> B0+B1x


pred4 = model4.predict(pd.DataFrame(cc))
pred4_at = np.exp(pred4)
pred4_at

#For visualization there is a problem in Python they can't visualize with 3 variable, only 2 sholud be there, here, y ~ X + I X^2 (3 variables)
# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = cc.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = cc.iloc[:, 1].values


plt.scatter(cc.Weightgained, np.log(cc.CaloriesConsumed))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = cc.CaloriesConsumed - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4 #240.82777570407256


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Simple Linear Regression model  232.833501 gives least error

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(cc, test_size = 0.3)

finalmodel = smf.ols('CaloriesConsumed ~ Weightgained', data = train).fit()
finalmodel.summary() #R-squared:0.880= it is more than 0.8 => Strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 1542.0583 +2.2224(Weightgained) -> B0+B1x

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))

# Model Evaluation on Test data
test_res = test.CaloriesConsumed - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse #287.01327884391727


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.CaloriesConsumed - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse #215.24102568049625

