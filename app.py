# gerekli kurulumlar
!pip install xgboost imbalanced-learn

# Google Drive bağlantısı yapmak için gereken kod
from google.colab import drive
drive.mount('/content/drive')

# ZIP dosyasını açmak için gereken kod
import zipfile
zip_path = 'zip dosyasının yolu'
extract_path = 'zip dosyasının çıkarılacağı dosyanın yolu'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


import pandas as pd

#ZIP içindeki bir CSV dosyasını okumak için gereken kod
df = pd.read_csv('dolandırıcılık_veri_seti.csv')
df.head()


import os

# Klasördeki tüm dosyaları listeleyen kod
dosyalar = os.listdir(extract_path)

# Sadece .csv uzantılı dosyaları filtrele
csv_dosyalar = [f for f in dosyalar if f.endswith('.csv')]

# Eğer en az bir csv dosyası varsa ilkini okur
if csv_dosyalar:
    csv_yolu = os.path.join(extract_path, csv_dosyalar[0])
    veriler = pd.read_csv(csv_yolu)
    print("Yüklenen dosya:", csv_dosyalar[0])
    print(veriler.head())
else:
    print("Klasörde .csv dosyası bulunamadı.")


# Kütüphaneler
import zipfile
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


df.info()# veri setindeki sütunlar hakkında bilgi verir
df.describe()# sayısal sütünlarla ilgili temel bilgileri gösterir
df.isnull().sum()# hangi sütunda kaç adet eksik sütun olduğunu gösterir

# dengesizlik olup olmadığını görmek için sütun grafiği
print("\n--- Sınıf Dağılımı ---")
sns.countplot(data=df, x='isFraud')
plt.title("Dolandırıcılık Dağılımı")
plt.show()


# gereksiz sütunları silme
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['type'], drop_first=True)


# modelin eğitileceği değişkenler
X = df.drop('isFraud', axis=1)
# modelin tahmin etmeyi öğreneceği değişkenler
y = df['isFraud']

# eğitim ve test veri setlerini bölme kodları
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


# eğitim seti üzerinde smote işlemi(veri dengesizliğini önlemek için)
print("SMOTE uygulanıyor...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE sonrası sınıf dağılımı:", y_train_resampled.value_counts().to_dict())

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
import numpy as np
import joblib

# Makine Öğrenimi Modelleri
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
}

# Kullanılacak doğruluk skorları
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score),
    "roc_auc": "roc_auc"
}

def evaluate_model_cv(model, X, y, cv=5):
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    results = {}
    for metric in scoring.keys():
        results[metric] = (np.mean(scores[f'test_{metric}']), np.std(scores[f'test_{metric}']))
    return results

print("\n--- Modeller 3-Fold Cross Validation ile Değerlendiriliyor ---")

best_model = None
best_model_name = None
best_f1_mean = 0.0

for name, model in models.items():
    print(f"\nModel: {name}")
    results = evaluate_model_cv(model, X_train_resampled, y_train_resampled, cv=3)
    print(f"Accuracy: {results['accuracy'][0]:.4f} ± {results['accuracy'][1]:.4f}")
    print(f"F1 Score: {results['f1'][0]:.4f} ± {results['f1'][1]:.4f}")
    print(f"ROC AUC: {results['roc_auc'][0]:.4f} ± {results['roc_auc'][1]:.4f}")

    # En iyi modeli F1 skoruna göre seçelim (dengesiz veri için önemli)
    if results['f1'][0] > best_f1_mean:
        best_f1_mean = results['f1'][0]
        best_model = model
        best_model_name = name

# En iyi modeli tüm eğitim verisiyle tekrar eğitmek için gereken kod
print(f"\n En iyi model: {best_model_name} (Cross-Validated F1: {best_f1_mean:.4f})")
best_model.fit(X_train_resampled, y_train_resampled)

# Modeli kaydetmek için:
joblib.dump(best_model, "/content/fraud_model.pkl")


# en iyi modelin detaylı değerlendirmesi
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print("\n SINIFLANDIRMA RAPORU")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
