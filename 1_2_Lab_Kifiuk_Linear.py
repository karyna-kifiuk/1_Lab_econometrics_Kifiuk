import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Дані
x = np.array([9.8, 10.2, 13.5, 15.2, 17.2, 19.6, 21.3, 23.5, 25.6, 26.3]).reshape((-1, 1))
y = np.array([7.8, 9.5, 12.3, 14.7, 17.1, 18.1, 20.8, 23.1, 23.8, 24.6])

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept: {new_model.intercept_}")


print(f"slope: {new_model.coef_}")

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")

x_new = np.arange(5).reshape((-1, 1))
x_new
y_new = model.predict(x_new)
y_new

# Побудова графіку
plt.scatter(x, y, color='b', label='Спостережені дані')
plt.plot(x, y_pred, color='r', linewidth=2, label='Лінійна регресія')
plt.xlabel('x (незалежна змінна)')
plt.ylabel('y (залежна змінна)')
plt.legend()
plt.show()
