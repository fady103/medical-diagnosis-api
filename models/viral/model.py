import pickle
import numpy as np

# إصدار النموذج
__version__ = "1.0"

# تحميل النموذج
with open("models/viral/Gradient_Boosting_Classifier_model_Viral_infection.pkl", "rb") as f:
    model = pickle.load(f)

def predict_pipeline(data: dict) -> str:
    """
    توقع الإصابة بعدوى فيروسية بناءً على البيانات المعطاة.
    """
    input_array = np.array([list(data.values())])
    prediction = model.predict(input_array)[0]
    return "Viral Infection" if prediction == 1 else "No Infection"
