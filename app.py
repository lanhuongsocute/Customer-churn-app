from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# ================== Load Mô Hình, Scaler Và LabelEncoder ==================
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoder.pkl')  # Load LabelEncoders

# ================== Tên Cột Dữ Liệu ==================
columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
           'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
           'PaymentMethod', 'MonthlyCharges', 'TotalCharges']


@app.route('/')
def index():
    """Trang chủ hiển thị giao diện nhập dữ liệu."""
    return render_template('index.html')


# ================== Xử Lý Dữ Liệu Người Dùng Nhập ==================
@app.route('/parse', methods=['POST'])
def parse():
    """Nhận dữ liệu đầu vào, tách thành các giá trị và trả về dưới dạng JSON."""
    try:
        # Lấy dữ liệu từ textarea
        input_text = request.form['input_text'].strip()

        # Kiểm tra dữ liệu: Nếu có nhiều dòng (CSV), chỉ lấy dòng đầu tiên sau header
        if '\n' in input_text:
            input_data = input_text.split('\n')[1].split(',')
        else:
            input_data = input_text.split(',')

        # ================== Kiểm Tra Số Lượng Cột ==================
        if len(input_data) != len(columns):
            return jsonify({'error': f"Số lượng cột không khớp. Dữ liệu đầu vào có {len(input_data)} cột, nhưng cần {len(columns)} cột."})

        # Tạo DataFrame từ dữ liệu đã tách
        data_df = pd.DataFrame([input_data], columns=columns)

        # Trả về dữ liệu dưới dạng JSON để hiển thị trên giao diện
        return jsonify(data_df.iloc[0].to_dict())

    except Exception as e:
        return jsonify({'error': str(e)})


# ================== Dự Đoán Churn ==================
@app.route('/predict', methods=['POST'])
def predict():
    """Nhận dữ liệu từ form, xử lý và trả về xác suất khách hàng Churn."""
    try:
        # Lấy dữ liệu từ form nhập liệu
        data = request.form.to_dict()

        # Tạo DataFrame với các giá trị từ form
        data_df = pd.DataFrame([list(data.values())], columns=columns)

        # ================== Xử Lý Dữ Liệu ==================
        # 1. Chuyển đổi các cột số sang kiểu float
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

        # 2. Mã hóa các cột phân loại bằng LabelEncoder
        for col in data_df.columns:
            if col in le_dict:  # Nếu cột là cột phân loại
                le = le_dict[col]
                data_df[col] = data_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                data_df[col] = le.transform(data_df[col].astype(str))

        # 3. Xử Lý Trường Hợp Cột Toàn Bộ Là NaN
        for col in numeric_cols:
            if data_df[col].isna().all():
                data_df[col].fillna(0, inplace=True)  # Gán giá trị 0 nếu cột toàn bộ là NaN

        # ================== Xử Lý Các Giá Trị NaN ==================
        imputer = SimpleImputer(strategy='median')
        data_df = pd.DataFrame(imputer.fit_transform(data_df), columns=data_df.columns)

        # ================== Chuẩn Hóa Dữ Liệu ==================
        data_df = pd.DataFrame(scaler.transform(data_df), columns=data_df.columns)

        # ================== Dự Đoán Xác Suất Churn ==================
        prediction = model.predict_proba(data_df)[:, 1][0] * 100

        return jsonify({'prediction': f'{prediction:.2f}%'})

    except Exception as e:
        return jsonify({'error': str(e)})


# ================== Chạy Ứng Dụng Flask ==================
if __name__ == '__main__':
    app.run(debug=True)
