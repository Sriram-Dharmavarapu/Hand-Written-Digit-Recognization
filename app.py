# app.py (improved preprocessing + streamlit demo)
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import os

# optional canvas
try:
    from streamlit_drawable_canvas import st_canvas
    HAVE_CANVAS = True
except Exception:
    HAVE_CANVAS = False

@st.cache_resource
def load_model_cached(path='digit_model.h5'):
    # load without compiling to avoid compile warnings for inference-only use
    return load_model(path, compile=False)

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Digit Recognizer — draw a digit (0–9)")

# load model (make sure digit_model.h5 is in same folder)
if not os.path.exists('digit_model.h5'):
    st.error("Model file 'digit_model.h5' not found. Put it in this folder.")
    st.stop()
model = load_model_cached('digit_model.h5')

# ---- preprocessing utilities ----
def prepare_image_for_model(pil_img):
    """
    Takes a PIL grayscale or RGB image and returns:
      - final_pil: processed 28x28 PIL image (for preview)
      - x: numpy array shaped (1,28,28,1) normalized 0..1 ready for model.predict
    Steps:
      1. convert to grayscale
      2. invert if background is white
      3. crop to bounding box of digit
      4. resize preserving aspect: largest side => 20 pixels
      5. pad to 28x28 centered
      6. center by mass (shift so centroid ~ (14,14))
    """
    # grayscale
    img = pil_img.convert('L')

    arr = np.array(img).astype(np.uint8)

    # If mostly light (white background), invert so that digit pixels are bright
    if arr.mean() > 127:
        arr = 255 - arr
    # threshold small values (reduce noise)
    thresh = 10
    mask = arr > thresh
    if not np.any(mask):
        # empty -> return blank image and zero input
        final = Image.new('L', (28,28), color=0)
        x = np.zeros((1,28,28,1), dtype=np.float32)
        return final, x

    # crop to content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = Image.fromarray(arr[rmin:rmax+1, cmin:cmax+1])

    # resize preserving aspect ratio: largest side -> 20
    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round((20 * h) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((20 * w) / h)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # pad to 28x28 centered
    new_img = Image.new('L', (28,28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))

    # compute center of mass and shift to center (14,14)
    np_img = np.array(new_img).astype(np.float32)
    total = np.sum(np_img)
    if total > 0:
        # centroid
        cx = np.sum(np.arange(28) * np.sum(np_img, axis=0)) / total
        cy = np.sum(np.arange(28) * np.sum(np_img, axis=1)) / total
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))
        shifted = np.roll(np_img, shift_y, axis=0)
        shifted = np.roll(shifted, shift_x, axis=1)
        # zero wrapped areas
        if shift_y > 0:
            shifted[:shift_y, :] = 0
        elif shift_y < 0:
            shifted[shift_y:, :] = 0
        if shift_x > 0:
            shifted[:, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0
        final_arr = shifted
    else:
        final_arr = np_img

    # normalize to 0..1 and reshape
    final_arr = np.clip(final_arr, 0, 255).astype(np.uint8)
    final_pil = Image.fromarray(final_arr, mode='L')
    x = final_arr.astype('float32') / 255.0
    x = x.reshape(1, 28, 28, 1)
    return final_pil, x

# ---- UI ----
st.write("Draw inside the box (or upload). The app will preprocess to 28×28 like MNIST.")

if HAVE_CANVAS:
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        pil_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
        processed_pil, x = prepare_image_for_model(pil_img)
        st.image(processed_pil.resize((140,140)), caption="Processed (28x28) preview", width=140)
        if st.button("Predict"):
            probs = model.predict(x)[0]
            pred = int(np.argmax(probs))
            st.success(f"Predicted digit: {pred}")
            st.bar_chart(probs)

else:
    st.info("Install 'streamlit-drawable-canvas' for drawing. Fallback: upload an image.")
    uploaded = st.file_uploader("Upload a digit image (png/jpg)", type=['png','jpg','jpeg'])
    if uploaded is not None:
        pil_uploaded = Image.open(uploaded)
        processed_pil, x = prepare_image_for_model(pil_uploaded)
        st.image(processed_pil.resize((140,140)), caption='Processed (28x28) preview', width=140)
        if st.button("Predict Upload"):
            probs = model.predict(x)[0]
            st.success(f"Predicted digit: {int(np.argmax(probs))}")
            st.bar_chart(probs)

st.write("---")
st.markdown("**Notes:**\n- Model expects single-digit images. For multi-digit images you must segment digits first.\n- If predictions still look wrong, see training tips below.")
