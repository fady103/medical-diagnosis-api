import pickle
import numpy as np

# إصدار النموذج
__version__ = "1.0"

# تحميل النموذج
with open("models/diabetes/random_forest_model_diapetes.pkl", "rb") as f:
    model = pickle.load(f)

# دالة التنبؤ
def predict_pipeline(data: dict) -> str:
    """
    يستقبل بيانات المريض كقاموس، ويرجع التشخيص باستخدام نموذج السكري.
    """
    input_array = np.array([list(data.values())])
    prediction = model.predict(input_array)[0]
    return "Diabetes" if prediction == 1 else "No Diabetes"
