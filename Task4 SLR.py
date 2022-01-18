# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

dt = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Simple Linear Regression/Salary_Data.csv")

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

plt.bar(height = dt.YearsExperience, x = np.arange(1, 31, 1)) #numpy.arange([start, ]stop, [step, ], dtype=None) -> numpy.ndarray
plt.hist(dt.YearsExperience) #histogram
plt.boxplot(dt.YearsExperience) #boxplot

plt.bar(height = dt.Salary, x = np.arange(1, 31, 1))
plt.hist(dt.Salary) #histogram
plt.boxplot(dt.Salary) #boxplot

# Scatter plot
plt.scatter(x = dt['Salary'], y = dt['YearsExperience'], color = 'green') 

# correlation
np.corrcoef(dt.Salary, dt.YearsExperience) #0.97824162 - strong Relation (greater than 0.85)

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(dt.Salary, dt.YearsExperience)[0, 1]
cov_output #76106.30344827585 = positive direction , strong positive correlation

# dt.cov()


# Import library
import statsmodels.formula.api as smf # api gives Ordinary Least Square - sum of squares,(y-y^)^2+..., best fit line

# Simple Linear Regression
model = smf.ols('Salary ~ YearsExperience', data = dt).fit()
model.summary() # p value should always less than 0.05 - if yes then it is statistically significant, 
#R-squared:0.957 = it is greater than 0.8 => strong correlation => goodness of fit
# Equation is, y = 2.579e+04+9449.9623(YearsExperience) -> B0+B1x

pred1 = model.predict(pd.DataFrame(dt['YearsExperience']))

# Regression Line
plt.scatter(dt.YearsExperience, dt.Salary)
plt.plot(dt.YearsExperience, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = dt.Salary - pred1 # Actual - Predicted
res_sqr1 = res1 * res1 # r square
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 #5592.043608760661

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(dt['YearsExperience']), y = dt['Salary'], color = 'brown')
np.corrcoef(np.log(dt.YearsExperience), dt.Salary) #correlation = 0.92406108 (greater than 0.85) => strong correlation

model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = dt).fit()
model2.summary() #R-squared:0.854 = it is greater than to 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 1.493e+04+4.058e+04(YearsExperience) -> B0+B1x

pred2 = model2.predict(pd.DataFrame(dt['YearsExperience']))

# Regression Line
plt.scatter(np.log(dt.YearsExperience), dt.Salary)
plt.plot(np.log(dt.YearsExperience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = dt.Salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 #10302.893706228308


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = dt['YearsExperience'], y = np.log(dt['Salary']), color = 'orange')
np.corrcoef(dt.YearsExperience, np.log(dt.Salary)) #correlation =  0.96538444 - strong Relation (greater than 0.85)

model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = dt).fit()
model3.summary() #R-squared: 0.932 = it is greater than 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 10.5074+0.1255(YearsExperience) -> B0+B1x


pred3 = model3.predict(pd.DataFrame(dt['YearsExperience']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(dt.YearsExperience, np.log(dt.Salary))
plt.plot(dt.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = dt.Salary- pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 #7213.235076620129


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = dt).fit() # y ~ X + I X^2 + I X^3
model4.summary() #R-squared: 0.949 = it is greater than 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 10.3369-0.2024(YearsExperience) -> B0+B1x


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


plt.scatter(dt.YearsExperience, np.log(dt.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = dt.Salary - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4 #5391.081582693624


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Polynomial model 5391.081583 gives least error

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(dt, test_size = 0.2)

finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = train).fit()
finalmodel.summary() #R-squared:0.948= it is greater than 0.8 => strong correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 10.3709+0.1828(YearsExperience) -> B0+B1x

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Salary = np.exp(test_pred)
pred_test_Salary

# Model Evaluation on Test data
test_res = test.Salary - pred_test_Salary
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse #7158.105418075264


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Salary = np.exp(train_pred)
pred_train_Salary

# Model Evaluation on train data
train_res = train.Salary - pred_train_Salary
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse #5283.737469408052

