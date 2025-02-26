import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ======================= Đọc và làm sạch dữ liệu =======================
data = pd.read_csv('Telcom-Customer-Churn.csv')

# Loại bỏ các cột không cần thiết
if 'customerID' in data.columns:
    data = data.drop(columns=['customerID'])

# Xử lý các giá trị thiếu
imputer = SimpleImputer(strategy='median')
data[data.select_dtypes(include=np.number).columns] = imputer.fit_transform(data.select_dtypes(include=np.number))

# ======================= Mã hóa các cột phân loại =======================
le_dict = {}  # Dictionary để lưu LabelEncoder cho từng cột
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le  # Lưu LabelEncoder cho từng cột

# Lưu LabelEncoders vào tệp label_encoder.pkl
joblib.dump(le_dict, 'label_encoder.pkl')

# ======================= Tách dữ liệu thành X và y =======================
X = data.drop(columns=['Churn'])
y = data['Churn']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Lưu scaler vào tệp scaler.pkl
joblib.dump(scaler, 'scaler.pkl')

# ======================= Chia dữ liệu thành tập huấn luyện và kiểm thử =======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ======================= Huấn luyện mô hình Logistic Regression =======================
model = LogisticRegression(max_iter=500, C=0.7, solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# ======================= Lưu mô hình vào tệp =======================
joblib.dump(model, 'churn_model.pkl')

print("✅ Mô hình, Scaler và LabelEncoder đã được lưu thành công!")
