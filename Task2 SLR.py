# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

dt = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Simple Linear Regression/delivery_time.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
dt.columns = "DeliveryTime", "SortingTime" ##renaming so that no sapces is there otherwise error.

dt.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = dt.DeliveryTime, x = np.arange(1, 22, 1)) #numpy.arange([start, ]stop, [step, ], dtype=None) -> numpy.ndarray
plt.hist(dt.DeliveryTime) #histogram
plt.boxplot(dt.DeliveryTime) #boxplot

plt.bar(height = dt.SortingTime, x = np.arange(1, 22, 1))
plt.hist(dt.SortingTime) #histogram
plt.boxplot(dt.SortingTime) #boxplot

# Scatter plot
plt.scatter(x = dt['DeliveryTime'], y = dt['SortingTime'], color = 'green') 

# correlation
np.corrcoef(dt.DeliveryTime, dt.SortingTime) #0.82599726 - moderate Relation (less than 0.85)

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(dt.DeliveryTime, dt.SortingTime)[0, 1]
cov_output #10.655809523809523

# dt.cov()


# Import library
import statsmodels.formula.api as smf # api gives Ordinary Least Square - sum of squares,(y-y^)^2+..., best fit line

# Simple Linear Regression
model = smf.ols('DeliveryTime ~ SortingTime', data = dt).fit()
model.summary() # p value should always less than 0.05 - if yes then it is statistically significant, 
#R-squared:0.682 = it is less than 0.8 => moderate correlation => goodness of fit
# Equation is, y = 6.5827+1.6490(SortingTime) -> B0+B1x

pred1 = model.predict(pd.DataFrame(dt['SortingTime']))

# Regression Line
plt.scatter(dt.SortingTime, dt.DeliveryTime)
plt.plot(dt.SortingTime, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = dt.DeliveryTime - pred1 # Actual - Predicted
res_sqr1 = res1 * res1 # r square
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 #2.7916503270617654

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(dt['SortingTime']), y = dt['DeliveryTime'], color = 'brown')
np.corrcoef(np.log(dt.SortingTime), dt.DeliveryTime) #correlation = 0.83393253 (less than 0.85)

model2 = smf.ols('DeliveryTime ~ np.log(SortingTime)', data = dt).fit()
model2.summary() #R-squared:0.695 = it is less than 0.8 => moderate correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 1.1597+9.0434(SortingTime) -> B0+B1x

pred2 = model2.predict(pd.DataFrame(dt['SortingTime']))

# Regression Line
plt.scatter(np.log(dt.SortingTime), dt.DeliveryTime)
plt.plot(np.log(dt.SortingTime), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = dt.DeliveryTime - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 #2.733171476682066


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = dt['SortingTime'], y = np.log(dt['DeliveryTime']), color = 'orange')
np.corrcoef(dt.SortingTime, np.log(dt.DeliveryTime)) #correlation =  0.84317726 - moderate Relation (greater than 0.85)

model3 = smf.ols('np.log(DeliveryTime) ~ SortingTime', data = dt).fit()
model3.summary() #R-squared: 0.711 = it is less than 0.8 => moderate correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 2.1214+0.1056(SortingTime) -> B0+B1x


pred3 = model3.predict(pd.DataFrame(dt['SortingTime']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(dt.SortingTime, np.log(dt.DeliveryTime))
plt.plot(dt.SortingTime, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = dt.DeliveryTime- pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 #2.940250323056201


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(DeliveryTime) ~ SortingTime + I(SortingTime*SortingTime)', data = dt).fit() # y ~ X + I X^2 + I X^3
model4.summary() #R-squared: 0.765 = it is less than 0.8 => moderate correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 1.6997+0.2659(SortingTime) -> B0+B1x


pred4 = model4.predict(pd.DataFrame(dt))
pred4_at = np.exp(pred4)
pred4_at

#For visualization there is a problem in Python they can't visualize with 3 variable, only 2 sholud be there, here, y ~ X + I X^2 (3 variables)
# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = dt.iloc[:, 1].values
X_poly = poly_reg.fit_transform(X)
# y = dt.iloc[:, 1].values


plt.scatter(dt.SortingTime, np.log(dt.DeliveryTime))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = dt.DeliveryTime - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4 #2.799041988740925


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Log Transformation model  2.733171 gives least error

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(dt, test_size = 0.2)

finalmodel = smf.ols('DeliveryTime ~ np.log(SortingTime)', data = train).fit()
finalmodel.summary() #R-squared:0.647= it is less than 0.8 => moderate correlation => goodness of fit
# p value should always less than 0.05 - if yes then it is statistically significant
# Equation is, y = 2.0886+8.3214(SortingTime) -> B0+B1x

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))

# Model Evaluation on Test data
test_res = test.DeliveryTime - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse #3.817383290493672


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.DeliveryTime - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse #2.350985338151398

