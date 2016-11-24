from sklearn import linear_model
data = linear_model.LinearRegression()
data.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

data.coef_
