# Dolandiricilik-tespiti


Bu proje, **XGBoost, Random Forest ve Logistic Regression** gibi makine öğrenimi algoritmaları kullanarak finansal işlemlerde **dolandırıcılık tespiti** yapmayı amaçlar.  
Model, **dengesiz veri setlerinde** daha başarılı sonuç almak için **SMOTE** yöntemi ile veri dengeleme uygular.

---

##  Özellikler
-  **Google Drive** üzerinden veri yükleme ve ZIP açma
-  **Veri analizi**: Eksik veri kontrolü, sınıf dağılımı, görselleştirme
-  **Makine öğrenimi modelleri**: Logistic Regression, Random Forest, XGBoost
-  **Cross-validation** ile model değerlendirme
-  En iyi modeli seçme ve kaydetme (`fraud_model.pkl`)
-  **Gradio arayüzü** ile web üzerinden kullanım (planlanabilir)

---



### Depoyu klonla
```bash
git clone https://github.com/kullanici_adi/fraud-detection.git
cd fraud-detection
