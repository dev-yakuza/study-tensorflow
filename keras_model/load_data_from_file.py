from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./data/score.csv', delimiter=',')
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

print('Data shape ======================================================')
print(f'x_data.shape: {x_data.shape}')
print(f'y_data.shape: {y_data.shape}')

model = Sequential()
model.add(Flatten(input_shape=(3,)))
model.add(Dense(1, activation='linear'))
# model.add(Dense(1, input_shape=(3,), activation='linear'))

model.compile(optimizer=SGD(learning_rate=1e-5), loss='mse')
model.summary()

hist = model.fit(x_data, y_data, epochs=1000, verbose=0)

predict_data = np.array([
    [5, 5, 0],
    [2, 3, 1],
    [-1, 0, -1],
    [-1, 0, -1],
    [10, 5, 2],
    [4, -1, -2],
])
result_data = [
    2 * data[0] - 3 * data[1] + 2 * data[2] for data in predict_data
]
result = model.predict(predict_data)

print('Result ==========================================================')
print(f'predict_data: {predict_data}')
print(f'result_data: {result_data}')
print(f'result: {result}')

print('Model info ======================================================')
print(model.input)
print(model.output)
print(model.weights)

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train_loss')
plt.legend(loc='best')

plt.show()