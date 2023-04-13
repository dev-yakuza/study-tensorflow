from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

x_data = np.array([1, 2, 3, 4, 5, 6])
y_data = np.array([3, 4, 5, 6, 7, 8])

print('Data shape ======================================================')
print(f'x_data.shape: {x_data.shape}')
print(f'y_data.shape: {y_data.shape}')

model = Sequential()
model.add(Flatten(input_shape=(1,)))
model.add(Dense(1, activation='linear'))

# Too big
# model.compile(optimizer=SGD(learning_rate=0.1), loss='mse')
# Too small
model.compile(optimizer=SGD(learning_rate=1e-5), loss='mse')
# Correct
# model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
# Better
# model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')
model.summary()

hist = model.fit(x_data, y_data, epochs=1000, verbose=1)

predict_data = np.array([-3.1, 3.0, 3.5, 15.0, 20.1])
result_data = np.array([-1.1, 5.0, 5.5, 17.0, 22.1])
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