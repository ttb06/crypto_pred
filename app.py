# Load mô hình TensorFlow
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("model_pred_1h.keras")

# Load StandardScaler
with open("scaler.pkl", "rb") as scaler_file:
    sc = pickle.load(scaler_file)

# Khởi tạo FastAPI
app = FastAPI()

# Định nghĩa schema đầu vào cho API
class ModelInput(BaseModel):
    data: list  # Danh sách số thực đầu vào

@app.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # Chuyển đổi dữ liệu đầu vào thành numpy array
        # input_array = np.array(input_data.data).reshape(1, -1)
        input_array = np.array(input_data.data)

        # Áp dụng StandardScaler để chuẩn hóa đầu vào
        # input_scaled = sc.transform(input_array)
        input_scaled = sc.transform(input_array.reshape((240, 1)))

        # Dự đoán với mô hình
        prediction_scaled = model.predict(input_scaled.reshape(1, 240, 1))

        # Đưa dự đoán về giá trị gốc bằng inverse_transform
        prediction = sc.inverse_transform(prediction_scaled)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
