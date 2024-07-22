from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Input, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Загрузка данных
features = np.load('features_augmented.npy')
labels = np.load('labels_augmented.npy')

# Преобразование данных для CNN-LSTM
X = np.array([feature.T for feature in features])
X = np.expand_dims(X, axis=-1)  # Добавление измерения для каналов
y = labels

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
x = LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='linear')(x)  # Для регрессии количества повторений

model = Model(inputs, outputs)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test))

# Сохранение модели
model.save('melody_repetition_model_cnn_lstm_regularized.keras')

# Визуализация обучения
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()