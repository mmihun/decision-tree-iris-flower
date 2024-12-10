# Import pustaka yang diperlukan
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. Memuat Dataset Iris dari sklearn
data = load_iris()
X = data.data  # Fitur (panjang/lebar kelopak dan mahkota)
y = data.target  # Label (setosa, versicolor, virginica)

# Menyimpan nama fitur dan label
feature_names = data.feature_names
target_names = data.target_names

# 2. Membagi Data Menjadi Pelatihan dan Pengujian dengan 2 Fitur
feature_idx1, feature_idx2 = 0, 1  # Menggunakan sepal length dan sepal width
X_train, X_test, y_train, y_test = train_test_split(
    X[:, [feature_idx1, feature_idx2]], y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nJumlah Data Pelatihan: {X_train.shape[0]}")
print(f"Jumlah Data Pengujian: {X_test.shape[0]}")

# 3. Membuat dan Melatih Model Decision Tree dengan 2 Fitur
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy:.2f}")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Menyimpan Model
joblib.dump(model, "decision_tree_iris.pkl")
print("\nModel telah disimpan sebagai 'decision_tree_iris.pkl'")

# 6. Memuat dan Menggunakan Model
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
        sample = np.array([[sepal_length, sepal_width]])
        prediction = loaded_model.predict(sample)
        print(f"\nPrediksi untuk sampel {sample}: {target_names[prediction[0]]}")
    else:
        print("\nError: Angka yang dimasukkan tidak sesuai dengan rentang yang diharapkan untuk dataset Iris.")
except ValueError:
    print("Harap masukkan angka yang valid untuk semua fitur.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")

# 7. Visualisasi Decision Boundaries dengan Hasil Prediksi
def plot_decision_boundaries(X, y, model, feature_idx1=0, feature_idx2=1, prediction=None, sample=None):
    # Plot area keputusan
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Prediksi untuk setiap titik dalam meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot area keputusan
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA']))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF', '#00FF00']),
                          edgecolor='k', marker='o')

    # Menambahkan legenda dengan label spesies
    handles, labels = scatter.legend_elements()
    plt.legend(handles, labels, title="Spesies", loc="upper right")

    # Menambahkan titik hasil prediksi
    if prediction is not None and sample is not None:
        # Menampilkan titik hasil prediksi
        plt.scatter(sample[:, 0], sample[:, 1], color='black', s=100, edgecolor='k', marker='x')
        # Menampilkan teks prediksi dengan bounding box untuk visibilitas
        plt.text(sample[:, 0], sample[:, 1], f'Prediksi: {target_names[prediction[0]]}', color='black',
                 fontsize=12, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.xlabel(feature_names[feature_idx1])
    plt.ylabel(feature_names[feature_idx2])
    plt.title("Decision Boundaries of Decision Tree with Prediction")
    plt.show()

# Plot decision boundaries menggunakan dua fitur pertama (sepal length dan sepal width)
plot_decision_boundaries(X[:, [feature_idx1, feature_idx2]], y, model, feature_idx1=feature_idx1, feature_idx2=feature_idx2, 
                         prediction=prediction, sample=sample)
