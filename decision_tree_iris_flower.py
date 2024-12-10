# Import pustaka yang diperlukan
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import joblib
import numpy as np

# 1. Memuat Dataset Iris dari sklearn
data = load_iris()
X = data.data  # Fitur (panjang/lebar kelopak dan mahkota)
y = data.target  # Label (setosa, versicolor, virginica)

# Menyimpan nama fitur dan label
feature_names = data.feature_names
target_names = data.target_names

# 2. Memeriksa Data
print("5 Baris Pertama Fitur:")
print(X[:5])
print("\n5 Baris Pertama Label:")
print(y[:5])

# 3. Membagi Data Menjadi Pelatihan dan Pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nJumlah Data Pelatihan: {X_train.shape[0]}")
print(f"Jumlah Data Pengujian: {X_test.shape[0]}")

# 4. Membuat dan Melatih Model Decision Tree
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy:.2f}")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 7. Menyimpan Model
joblib.dump(model, "decision_tree_iris.pkl")
print("\nModel telah disimpan sebagai 'decision_tree_iris.pkl'")

# 8. Memuat dan Menggunakan Model
print("\nMembuka model yang telah disimpan...")
loaded_model = joblib.load("decision_tree_iris.pkl")

print("\nMasukkan data fitur bunga iris untuk prediksi:")
try:
    sepal_length = float(input("Panjang sepal (cm): "))
    sepal_width = float(input("Lebar sepal (cm): "))
    petal_length = float(input("Panjang petal (cm): "))
    petal_width = float(input("Lebar petal (cm): "))

    # Validasi input sesuai dengan rentang nilai dataset Iris
    if (0 <= sepal_length <= 10 and
        0 <= sepal_width <= 5 and
        0 <= petal_length <= 7 and
        0 <= petal_width <= 3.5):
        
        # Membuat prediksi menggunakan model yang telah dimuat
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = loaded_model.predict(sample)
        print(f"\nPrediksi untuk sampel {sample}: {target_names[prediction[0]]}")
    else:
        print("\nError: Angka yang dimasukkan tidak sesuai dengan rentang yang diharapkan untuk dataset Iris.")
except ValueError:
    print("Harap masukkan angka yang valid untuk semua fitur.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
