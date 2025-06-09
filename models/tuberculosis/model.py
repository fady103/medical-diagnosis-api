from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# إصدار النموذج
__version__ = "1.0"

# تحميل النموذج
model = load_model("models/tuberculosis/CNN_Tuberculosis_model.h5")

def predict_pipeline(img_path: str) -> str:
    """
    تحميل صورة وتحليلها باستخدام نموذج الكشف عن السل (Tuberculosis).
    """
    img = image.load_img(img_path, color_mode="grayscale", target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    return "Tuberculosis" if prediction >= 0.5 else "Normal"
