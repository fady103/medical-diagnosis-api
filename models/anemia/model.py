import pickle
import numpy as np

# إصدار النموذج
__version__ = "1.0"

# تحميل النموذج
with open("models/anemia/Logistic_Regression_model_Anemia.pkl", "rb") as f:
    model = pickle.load(f)

def predict_pipeline(data: dict) -> str:
    """
    يستقبل بيانات المريض كقاموس، ويقوم بتشخيص حالة فقر الدم.
    """
    input_array = np.array([list(data.values())])
    prediction = model.predict(input_array)[0]
    return "Anemia" if prediction == 1 else "Normal"
