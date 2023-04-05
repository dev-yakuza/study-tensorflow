from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./data/diabetes.csv', delimiter=',', skiprows=1)
x_data = data[0:, 0:-1]
y_data = data[0:, [-1]]

print('Data shape ======================================================')
print(f'x_data.shape: {x_data.shape}')
print(f'y_data.shape: {y_data.shape}')

model = Sequential()
model.add(Flatten(input_shape=(x_data.shape[1],)))
model.add(Dense(y_data.shape[1], activation='sigmoid'))
# model.add(Dense(y_data.shape[1], input_shape=(x_data.shape[1],), activation='sigmoid'))

model.compile(optimizer=SGD(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_data, y_data, epochs=500, validation_split=0.2, verbose=0)

predict_data = np.array([
    x_data[0],
    x_data[1],
])
result_data = [
    y_data[0],
    y_data[1],
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

print('Loss Trend ======================================================')
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='validation_loss')
plt.legend(loc='best')

plt.show()

print('Accuracy ======================================================')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train_accuracy')
plt.plot(hist.history['val_accuracy'], label='validation_accuracy')
plt.legend(loc='best')

plt.show()