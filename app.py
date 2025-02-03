# Load mô hình TensorFlow
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import joblib

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("model_pred_1h_3output.keras")

# Load StandardScaler
scalers = []
for i in range(3):
    name = f'scaler_{i}.pkl'
    # sc = pickle.load(open(name, 'rb'))
    sc = joblib.load(f'scaler_{i}.pkl')
    scalers.append(sc)

# Khởi tạo FastAPI
app = FastAPI()
checker = 0
# Định nghĩa schema đầu vào cho API
class ModelInput(BaseModel):
    data: list  # Danh sách số thực đầu vào

@app.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # Chuyển đổi dữ liệu đầu vào thành numpy array
        # input_array = np.array(input_data.data).reshape(1, -1)
        input_array = np.array(input_data.data)
        # return(input_array)
        input_array = input_array.reshape(240, 1, 3)

        # print(input_array.shape)

        # Áp dụng StandardScaler để chuẩn hóa đầu vào
        # input_scaled = sc.transform(input_array)

        # input_scaled = sc.transform(input_array.reshape((240, 1)))
        for i in range(3):
            input_scaled = scalers[i].transform(input_array[:, :, i])
            input_array[:, :, i] = input_scaled.reshape(-1, 1)

        # Dự đoán với mô hình
        prediction_scaled = model.predict(input_array.reshape(1, 240, 1, 3))

        # Đưa dự đoán về giá trị gốc bằng inverse_transform
        # prediction = sc.inverse_transform(prediction_scaled)
        predictions = []
        for i in range(3):
            prediction = scalers[i].inverse_transform(prediction_scaled[:, :, i])
            print(prediction_scaled)
            predictions.append(prediction[0].tolist()[0])

        return {"prediction": predictions}
        # print(predictions)
    except Exception as e:
        # print(e)
        return {"error": str(e), "shape": input_array.shape}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
