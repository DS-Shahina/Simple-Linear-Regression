# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

dt = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Simple Linear Regression/SAT_GPA.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

dt.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = dt.SAT_Scores, x = np.arange(1, 201, 1)) #numpy.arange([start, ]stop, [step, ], dtype=None) -> numpy.ndarray
plt.hist(dt.SAT_Scores) #histogram
plt.boxplot(dt.SAT_Scores) #boxplot

plt.bar(height = dt.GPA, x = np.arange(1, 201, 1))
plt.hist(dt.GPA) #histogram
plt.boxplot(dt.GPA) #boxplot

# Scatter plot
plt.scatter(x = dt['SAT_Scores'], y = dt['GPA'], color = 'green') 

# correlation
np.corrcoef(dt.GPA, dt.SAT_Scores) #0.29353828 - weak Relation (less than 0.85)

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(dt.GPA, dt.SAT_Scores)[0, 1]
cov_output #27.777793969849263 = positive direction , weak positive correlation

# dt.cov()


# Import library
import statsmodels.formula.api as smf # api gives Ordinary Least Square - sum of squares,(y-y^)^2+..., best fit line

# Simple Linear Regression
model = smf.ols('GPA ~ SAT_Scores', data = dt).fit()
model.summary() # p value should always less than 0.05 - if yes then it is statistically significant, 
#R-squared:0.086 = it is less than 0.8 => weak correlation => goodness of fit
# Equation is, y = 2.4029+0.0009(SAT_Scores) -> B0+B1x

pred1 = model.predict(pd.DataFrame(dt['SAT_Scores']))

# Regression Line
plt.scatter(dt.SAT_Scores, dt.GPA)
plt.plot(dt.SAT_Scores, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = dt.GPA - pred1 # Actual - Predicted
res_sqr1 = res1 * res1 # r square
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 #0.5159457227723683

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(dt['SAT_Scores']), y = dt['GPA'], color = 'brown')
np.corrcoef(np.log(dt.SAT_Scores), dt.GPA) #correlation = 0.27771976 (less than 0.85) => weak correlation

model2 = smf.ols('GPA ~ np.log(SAT_Scores)', data = dt).fit()
model2.summary() #R-squared:0.077 = it is less than to 0.8 => weak correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 0.4796+0.3868(SAT_Scores) -> B0+B1x

pred2 = model2.predict(pd.DataFrame(dt['SAT_Scores']))

# Regression Line
plt.scatter(np.log(dt.SAT_Scores), dt.GPA)
plt.plot(np.log(dt.SAT_Scores), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = dt.GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 #0.5184904101080668


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = dt['SAT_Scores'], y = np.log(dt['GPA']), color = 'orange')
np.corrcoef(dt.SAT_Scores, np.log(dt.GPA)) #correlation =  0.29408419 - weak Relation (less than 0.85)

model3 = smf.ols('np.log(GPA) ~ SAT_Scores', data = dt).fit()
model3.summary() #R-squared: 0.086 = it is less than 0.8 => weak correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 0.8727+0.0003(SAT_Scores) -> B0+B1x


pred3 = model3.predict(pd.DataFrame(dt['SAT_Scores']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(dt.SAT_Scores, np.log(dt.GPA))
plt.plot(dt.SAT_Scores, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = dt.GPA- pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 #0.5175875893834132


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = dt).fit() # y ~ X + I X^2 + I X^3
model4.summary() #R-squared: 0.094 = it is less than 0.8 => weak correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 1.0056-0.0003(SAT_Scores) -> B0+B1x


pred4 = model4.predict(pd.DataFrame(dt))
pred4_at = np.exp(pred4)
pred4_at

#For visualization there is a problem in Python they can't visualize with 3 variable, only 2 sholud be there, here, y ~ X + I X^2 (3 variables)
# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = dt.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = dt.iloc[:, 1].values


plt.scatter(dt.SAT_Scores, np.log(dt.GPA))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = dt.GPA - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4 #0.5144912487746159


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Polynomial model 0.514491 gives least error

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(dt, test_size = 0.2)

finalmodel = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = train).fit()
finalmodel.summary() #R-squared:0.114= it is less than 0.8 => weak correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 1.0867-0.0006(SAT_Scores) -> B0+B1x

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_GPA = np.exp(test_pred)
pred_test_GPA

# Model Evaluation on Test data
test_res = test.GPA - pred_test_GPA
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse #0.5024200350455968


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_GPA = np.exp(train_pred)
pred_train_GPA

# Model Evaluation on train data
train_res = train.GPA - pred_train_GPA
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse #0.5159107649628971

