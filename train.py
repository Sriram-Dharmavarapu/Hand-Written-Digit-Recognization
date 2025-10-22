# train.py
import argparse
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

def build_model(input_shape=(28,28,1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(epochs=5, batch_size=128, model_path='digit_model.h5'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    model = build_model()
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}%")
    model.save(model_path)
    print(f"Saved model to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='digit_model.h5')
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.model_path)
