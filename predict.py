# predict.py
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def preprocess_image(path):
    img = Image.open(path).convert('L')  # grayscale
    # resize first to keep aspect then center-crop logic simple
    img = img.resize((28, 28), Image.ANTIALIAS)
    arr = np.array(img).astype('float32')

    # If the image background is white and digit dark, invert so digit is white on black (like MNIST)
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def predict(image_path, model_path='digit_model.h5'):
    model = load_model(model_path)
    x = preprocess_image(image_path)
    probs = model.predict(x)[0]
    pred = int(np.argmax(probs))
    return pred, probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file (png/jpg)')
    parser.add_argument('--model', default='digit_model.h5', help='Path to saved model')
    args = parser.parse_args()

    pred, probs = predict(args.image, args.model)
    print(f"Predicted digit: {pred}")
    print("Probabilities:", np.round(probs, 4))
