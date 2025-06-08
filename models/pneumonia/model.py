from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# إصدار النموذج
__version__ = "1.0"

# تحميل النموذج
model = load_model("models/pneumonia/CNN_pneumonia_model.h5")

def predict_image(img_path: str) -> str:
    """
    يقوم بتحميل الصورة، تجهيزها، وتشغيل التنبؤ باستخدام نموذج الالتهاب الرئوي.
    """
    # تحميل الصورة كـ grayscale وتغيير الحجم
    img = image.load_img(img_path, color_mode="grayscale", target_size=(150, 150))
    
    # تحويل الصورة إلى مصفوفة أرقام وتطبيعها
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ بالنتيجة
    prediction = model.predict(img_array)[0][0]

    # إرجاع التصنيف كنص
    return "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
