# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

dt = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Simple Linear Regression/emp_data.csv")

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

plt.bar(height = dt.Salary_hike, x = np.arange(1, 11, 1)) #numpy.arange([start, ]stop, [step, ], dtype=None) -> numpy.ndarray
plt.hist(dt.Salary_hike) #histogram
plt.boxplot(dt.Salary_hike) #boxplot

plt.bar(height = dt.Churn_out_rate, x = np.arange(1, 11, 1))
plt.hist(dt.Churn_out_rate) #histogram
plt.boxplot(dt.Churn_out_rate) #boxplot

# Scatter plot
plt.scatter(x = dt['Churn_out_rate'], y = dt['Salary_hike'], color = 'green') 

# correlation
np.corrcoef(dt.Churn_out_rate, dt.Salary_hike) #-0.91172162 - strong negative Relation

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(dt.Churn_out_rate, dt.Salary_hike)[0, 1]
cov_output #-861.2666666666667 = negative direction , strong negative correlation

# dt.cov()


# Import library
import statsmodels.formula.api as smf # api gives Ordinary Least Square - sum of squares,(y-y^)^2+..., best fit line

# Simple Linear Regression
model = smf.ols('Churn_out_rate ~ Salary_hike', data = dt).fit()
model.summary() # p value should always less than 0.05 - if yes then it is statistically significant, 
#R-squared:0.831 = it is somewhat equal to 0.8 => moderate correlation => goodness of fit
# Equation is, y = 244.3649-0.1015(Salary_hike) -> B0+B1x

pred1 = model.predict(pd.DataFrame(dt['Salary_hike']))

# Regression Line
plt.scatter(dt.Salary_hike, dt.Churn_out_rate)
plt.plot(dt.Salary_hike, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = dt.Churn_out_rate - pred1 # Actual - Predicted
res_sqr1 = res1 * res1 # r square
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 #3.997528462337793

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(dt['Salary_hike']), y = dt['Churn_out_rate'], color = 'brown')
np.corrcoef(np.log(dt.Salary_hike), dt.Churn_out_rate) #correlation = -0.92120773 (greater than 0.85) => strong correlation

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data = dt).fit()
model2.summary() #R-squared:0.849 = it is somewhat equal to 0.8 => moderate correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 1381.4562-176.1097(Salary_hike) -> B0+B1x

pred2 = model2.predict(pd.DataFrame(dt['Salary_hike']))

# Regression Line
plt.scatter(np.log(dt.Salary_hike), dt.Churn_out_rate)
plt.plot(np.log(dt.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = dt.Churn_out_rate - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 #3.7860036130227708


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = dt['Salary_hike'], y = np.log(dt['Churn_out_rate']), color = 'orange')
np.corrcoef(dt.Salary_hike, np.log(dt.Churn_out_rate)) #correlation =  -0.93463607 - strong Relation (greater than 0.85)

model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data = dt).fit()
model3.summary() #R-squared: 0.874 = it is greater than 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 6.6383-0.0014(Salary_hike) -> B0+B1x


pred3 = model3.predict(pd.DataFrame(dt['Salary_hike']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(dt.Salary_hike, np.log(dt.Churn_out_rate))
plt.plot(dt.Salary_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = dt.Churn_out_rate- pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 #3.5415493188215756


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = dt).fit() # y ~ X + I X^2 + I X^3
model4.summary() #R-squared: 0.984 = it is greater than 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 23.1762-0.0207(Salary_hike) -> B0+B1x


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


plt.scatter(dt.Salary_hike, np.log(dt.Churn_out_rate))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = dt.Churn_out_rate - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4 #1.3267899683869546


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Polynomial model 1.326790 gives least error

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(dt, test_size = 0.2)

finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = train).fit()
finalmodel.summary() #R-squared:0.987= it is greater than 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 23.5009-0.0210(Salary_hike) -> B0+B1x

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Churn_out_rate = np.exp(test_pred)
pred_test_Churn_out_rate

# Model Evaluation on Test data
test_res = test.Churn_out_rate - pred_test_Churn_out_rate
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse #1.2365873085294992


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Churn_out_rate = np.exp(train_pred)
pred_train_Churn_out_rate

# Model Evaluation on train data
train_res = train.Churn_out_rate - pred_train_Churn_out_rate
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse #1.336095618357392

