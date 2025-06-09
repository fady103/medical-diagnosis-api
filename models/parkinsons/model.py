import pickle
import numpy as np

# إصدار النموذج
__version__ = "1.0"

# تحميل النموذج
with open("models/parkinsons/random_forest_model_Parkinsons.pkl", "rb") as f:
    model = pickle.load(f)

def predict_pipeline(data: dict) -> str:
    """
    يستقبل بيانات مريض ويُرجع تشخيص مرض باركنسون.
    """
    input_array = np.array([list(data.values())])
    prediction = model.predict(input_array)[0]
    return "Parkinson's Disease" if prediction == 1 else "Healthy"
