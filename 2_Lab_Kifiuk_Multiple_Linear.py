import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

#Дані
x = np.array([
    [41.8, 7.2, 35.5], [44.6, 9.2, 34.4], [42.0, 11.6, 30.5],
    [49.3, 13.4, 36.5], [48.6, 13.4, 32.1], [54.2, 14.8, 40.1],
    [62.2, 15.4, 34.1], [57.3, 15.8, 39.3], [54.1, 16.2, 39.2],
    [69.4, 16.5, 41.1], [60.2, 17.0, 42.5], [65.2, 17.1, 45.2], 
    [70.1, 18.0, 45.8], [75.5, 18.5, 43.9], [74.9, 19.0, 50.5], 
    [72.3, 20.5, 48.3]
])

y = np.array([4.50, 5.15, 6.00, 5.55, 5.70, 6.55, 5.90, 6.15, 6.95, 6.40,
 6.90, 7.35, 7.80, 8.00, 8.80, 9.30])

model = LinearRegression()
model.fit(x, y)

print("Коефіцієнти лінійної регресії:")
print(f"Коефіцієнт перетину (intercept): {model.intercept_}")
print(f"Коефіцієнти нахилу (coefficients): {model.coef_}")

# Графік для двох незалежних змінних та залежної
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, c='r', marker='o')

# Побудова площини регресії
xx, yy = np.meshgrid(x[:, 0], x[:, 1])
zz = model.intercept_ + model.coef_[0] * xx + model.coef_[1] * yy
ax.plot_surface(xx, yy, zz, color='b', alpha=0.5)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.show()