import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE, ADASYN
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

# Joblib uyarılarını sustur
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Veriyi yükle
dosya_yolu = r"C:\Users\Gözde\Desktop\masaustu_data_kaggle.csv"
df = pd.read_csv(dosya_yolu)

# Aykırı değerleri temizle
iso_forest = IsolationForest(contamination=0.03, random_state=42)
outliers = iso_forest.fit_predict(df[['Fiyat']])
df = df[outliers == 1]

# Gereksiz sütunları çıkar
drop_columns = ['Çözünürlük', 'Power Supply', 'Cihaz Ağırlığı', 'İşlemci Frekansı', 
                'Ekran Yenileme Hızı', 'Panel Tipi', 'Menşei', 'Ekran Boyutu']
df = df.drop(columns=drop_columns)

# Sayısal temizleme fonksiyonu
def clean_numeric(col):
    if col.dtype == 'object':
        col = col.str.replace('TB', '000 GB').str.replace('GB', '').str.replace('SSD Yok', '0').str.replace(r'[^0-9.]', '', regex=True)
        return pd.to_numeric(col, errors='coerce').fillna(0)
    return col

# Sayısal sütunları temizle ve doldur
df['SSD Kapasitesi'] = clean_numeric(df['SSD Kapasitesi'])
df['Kapasite'] = clean_numeric(df['Kapasite'])
df['Ram (Sistem Belleği)'] = clean_numeric(df['Ram (Sistem Belleği)'])
df['Temel İşlemci Hızı (GHz)'] = clean_numeric(df['Temel İşlemci Hızı (GHz)'])
df['Ekran Kartı Hafızası'] = clean_numeric(df['Ekran Kartı Hafızası'])
df['İşlemci Çekirdek Sayısı'] = clean_numeric(df['İşlemci Çekirdek Sayısı'])

# Yeni özellik: Bellek Kapasite Çarpımı ve Log Dönüşümü
df['Bellek Kapasite Çarpımı'] = df['Ram (Sistem Belleği)'] * df['Ekran Kartı Hafızası']
df['Bellek Kapasite Çarpımı_Log'] = np.log1p(df['Bellek Kapasite Çarpımı'])

# İşlemci Nesli'ni sayısal ve gruplu dönüştürelim
def nesil_to_numeric(nesil):
    if pd.isna(nesil) or isinstance(nesil, float):
        return 0
    if not isinstance(nesil, str):
        return 0
    if 'Nesil' in nesil:
        try:
            return int(nesil.split('.')[0])
        except (ValueError, IndexError):
            return 0
    return 0

df['İşlemci Nesli'] = df['İşlemci Nesli'].fillna('Belirtilmemiş')
df['İşlemci Nesli Sayısal'] = df['İşlemci Nesli'].apply(nesil_to_numeric)

# Diğer sayısal sütunları medyan ile doldur
df['Ram (Sistem Belleği)'] = df['Ram (Sistem Belleği)'].fillna(df['Ram (Sistem Belleği)'].median())
df['Temel İşlemci Hızı (GHz)'] = df['Temel İşlemci Hızı (GHz)'].fillna(df['Temel İşlemci Hızı (GHz)'].median())
df['Ekran Kartı Hafızası'] = df['Ekran Kartı Hafızası'].fillna(df['Ekran Kartı Hafızası'].median())
df['İşlemci Çekirdek Sayısı'] = df['İşlemci Çekirdek Sayısı'].fillna(df['İşlemci Çekirdek Sayısı'].median())

# Kategorik sütunlar
kategorik_sutunlar = ['Marka', 'İşlemci Tipi', 'Ekran Kartı', 'Kapasite', 'İşletim Sistemi', 
                      'Ekran Kartı Bellek Tipi', 'Ekran Kartı Tipi', 'Garanti Tipi', 
                      'Ram (Sistem Belleği) Tipi', 'İşlemci Nesli', 'İşlemci Modeli', 
                      'Kullanım Amacı', 'Bağlantılar', 'Arttırılabilir Azami Bellek']
df[kategorik_sutunlar] = df[kategorik_sutunlar].fillna('Belirtilmemiş')

# Kategorik verileri kodla
df_encoded = pd.get_dummies(df, columns=kategorik_sutunlar, drop_first=True)

# Özellikler (X) ve hedef (y)
X = df_encoded.drop('Fiyat', axis=1)
y = df_encoded['Fiyat']

# Yeni oran bazlı özellikler
X['Fiyat_Ram_Oran'] = df['Fiyat'] / (df['Ram (Sistem Belleği)'] + 1)
X['Fiyat_Cekirdek_Oran'] = df['Fiyat'] / (df['İşlemci Çekirdek Sayısı'] + 1)
X['Fiyat_SSD_Oran'] = df['Fiyat'] / (df['SSD Kapasitesi'] + 1)

# Fiyat aralıklarını yüzde 33 ve yüzde 66'lık dilimlere göre belirle (eski hali gibi)
bins = [df['Fiyat'].min(), df['Fiyat'].quantile(0.33), df['Fiyat'].quantile(0.66), df['Fiyat'].max()]
labels = [0, 1, 2]  # 0: Düşük, 1: Orta, 2: Yüksek
y_clf = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

# Özellik seçimi: En iyi 20 özellik (SVM ve LR için)
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y_clf)
selected_features = X.columns[selector.get_support(indices=True)]

# Eğitim/test ayırımı
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
X_train_sel, X_test_sel, _, _ = train_test_split(X_selected, y_clf, test_size=0.25, random_state=42, stratify=y_clf)

# Veriyi ölçeklendir (SVM ve KNN için)
scaler = StandardScaler()
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_full_scaled = scaler.transform(X_test_full)

# ADASYN yerine SMOTE kullan
smote = SMOTE(random_state=42, sampling_strategy={0: 610, 1: 900, 2: 610}, k_neighbors=5)
X_train_full_scaled, y_train_full = smote.fit_resample(X_train_full_scaled, y_train)
X_train_sel_scaled, y_train_sel = smote.fit_resample(X_train_sel_scaled, y_train)

# Sınıf dağılımını kontrol et
print("\nSınıf dağılımı:")
print("Orijinal veri:", pd.Series(y_train).value_counts().sort_index())
print("SMOTE sonrası:", pd.Series(y_train_full).value_counts().sort_index())

# Hiperparametre aramaları
param_grids = {
    "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['lbfgs'], 'max_iter': [5000], 'class_weight': ['balanced']},
    "Random Forest": {
        'n_estimators': [20, 30, 40],
        'max_depth': [3, 4, 5],
        'class_weight': ['balanced', 'balanced_subsample'],
        'min_samples_split': [10, 15],
        'min_samples_leaf': [4, 6]
    },
    "XGBoost": {
        'max_depth': [2],
        'learning_rate': [0.01, 0.02],
        'n_estimators': [30],
        'subsample': [0.6, 0.7],
        'colsample_bytree': [0.6, 0.7],
        'reg_alpha': [5, 10],
        'reg_lambda': [5, 10]
    },
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'probability': [True], 'class_weight': ['balanced']},
    "KNN": {'n_neighbors': [3, 5, 7]}
}

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier()
}

best_models = {}
accuracies = {}
errors = {}

for name, model in models.items():
    try:
        print(f"\n{name} için GridSearch başlatılıyor...")
        if name in ["SVM", "Logistic Regression", "KNN"]:
            grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_train_sel_scaled, y_train_sel)
            best_models[name] = grid.best_estimator_
            X_test_use = X_test_sel_scaled
            y_test_use = y_test
        else:
            grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_train_full_scaled, y_train_full)
            best_models[name] = grid.best_estimator_
            X_test_use = X_test_full_scaled
            y_test_use = y_test
        y_pred = best_models[name].predict(X_test_use)
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test_use, y_pred)
        f1 = f1_score(y_test_use, y_pred, average='macro')
        accuracies[name] = (acc, f1)
        print(f"{name} doğruluk: {acc:.4f}, f1_macro: {f1:.4f}")
    except Exception as e:
        errors[name] = str(e)
        print(f"{name} hata verdi: {e}")

print("\nModel doğrulukları ve f1_macro skorları:")
for name, (acc, f1) in accuracies.items():
    print(f"{name}: accuracy={acc:.4f}, f1_macro={f1:.4f}")
if errors:
    print("\nHata veren modeller:")
    for name, err in errors.items():
        print(f"{name}: {err}")

# Random Forest için çapraz doğrulama
rf_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf_model, X_train_full_scaled, y_train_full, cv=5, scoring='f1_macro')
print(f"Random Forest Çapraz Doğrulama F1-Macro Skorları: {cv_scores}")
print(f"Ortalama F1-Macro: {cv_scores.mean():.4f}, Standart Sapma: {cv_scores.std():.4f}")

# Excel için sonuçları saklayacak liste
excel_data = []

# ROC eğrileri için verileri saklayacak sözlükler
fpr_all = {0: {}, 1: {}, 2: {}}  # Sınıf bazında FPR
tpr_all = {0: {}, 1: {}, 2: {}}  # Sınıf bazında TPR
roc_auc_all = {0: {}, 1: {}, 2: {}}  # Sınıf bazında AUC

# Sınıf etiketlerini binarize et (ROC için)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

# Modelleri eğit ve değerlendir
for name, model in best_models.items():
    model.fit(X_train_full_scaled, y_train_full)
    y_pred = model.predict(X_test_full_scaled)
    y_prob = model.predict_proba(X_test_full_scaled) if hasattr(model, "predict_proba") else None
    
    # Performans metrikleri
    print(f"\n{name} Sınıflandırma Metrikleri:")
    report = classification_report(y_test, y_pred, target_names=['Düşük', 'Orta', 'Yüksek'], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['Düşük', 'Orta', 'Yüksek']))
    
    # Genel doğruluk (accuracy) değerini al
    accuracy = report['accuracy']
    
    # Metrikleri Excel için kaydet
    for cls in ['Düşük', 'Orta', 'Yüksek']:
        excel_data.append({
            'Model': name,
            'Sınıf': cls,
            'Accuracy': accuracy,
            'Precision': report[cls]['precision'],
            'Recall': report[cls]['recall'],
            'F1-Score': report[cls]['f1-score'],
            'Support': report[cls]['support']
        })
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Düşük', 'Orta', 'Yüksek'], yticklabels=['Düşük', 'Orta', 'Yüksek'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(f'Confusion Matrix ({name})')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.show()
    
    # ROC verilerini hesapla
    if y_prob is not None:
        for i in range(n_classes):
            fpr_all[i][name], tpr_all[i][name], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc_all[i][name] = auc(fpr_all[i][name], tpr_all[i][name])

# Sınıf bazında ROC eğrilerini aynı zeminde çiz (Zıt Renklerle)
class_names = ['Düşük', 'Orta', 'Yüksek']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Zıt renkler
model_names = list(best_models.keys())

for i in range(n_classes):
    plt.figure(figsize=(10, 6))  # Grafik boyutunu artır
    for j, (name, color) in enumerate(zip(model_names, colors)):
        if name in fpr_all[i]:
            plt.plot(fpr_all[i][name], tpr_all[i][name], color=color, lw=2, linestyle='-',
                     label=f'{name} (AUC = {roc_auc_all[i][name]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Rastgele Tahmin')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Eğrisi - Sınıf: {class_names[i]}', fontsize=14)
    
    # Legend ayarlarını optimize et
    plt.legend(loc="lower right", fontsize=8, framealpha=0.8, 
               bbox_to_anchor=(0.98, 0.02), borderaxespad=0.5)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'roc_curve_class_{i}_colors.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

# Özellik Önem Grafikleri
for name, model in best_models.items():
    if name in ["Random Forest", "XGBoost"]:
        feature_importance = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)

        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='Blues', legend=False)
        plt.title(f'En Önemli 10 Özellik ({name})', fontsize=12)
        plt.xlabel('Önem Skoru', fontsize=10)
        plt.ylabel('Özellik', fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}_updated.png', dpi=300)
        plt.show()

# PCA ile boyut indirgeme (2 boyuta düşürelim)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_full_scaled)
X_test_pca = pca.transform(X_test_full_scaled)

# Logistic Regression modelini PCA verisiyle tekrar eğit
lr_pca = LogisticRegression(max_iter=5000, random_state=42, solver='lbfgs')
lr_pca.fit(X_train_pca, y_train_full)

# Karar sınırı grafiği
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = lr_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Blues')
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', cmap='viridis')
plt.xlabel('Birinci Ana Bileşen (PCA 1)')
plt.ylabel('İkinci Ana Bileşen (PCA 2)')
plt.title('Logistic Regression Karar Sınırı (PCA İle)')
plt.legend(handles=scatter.legend_elements()[0], labels=['Düşük', 'Orta', 'Yüksek'], title='Fiyat Sınıfı')
plt.savefig('logistic_decision_boundary_updated.png', dpi=300)
plt.show()

# XGBoost için kayıp grafiği
eval_set = [(X_train_full_scaled, y_train_full), (X_test_full_scaled, y_test)]
xgb_model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train_full_scaled, y_train_full, eval_set=eval_set, verbose=False)

# Kayıp grafiği
evals_result = xgb_model.evals_result()
epochs = len(evals_result['validation_0']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 6))
plt.plot(x_axis, evals_result['validation_0']['mlogloss'], label='Eğitim Kaybı', color='blue')
plt.plot(x_axis, evals_result['validation_1']['mlogloss'], label='Doğrulama Kaybı', color='orange', linestyle='--')
plt.xlabel('İterasyon Sayısı')
plt.ylabel('Log-Loss')
plt.title('XGBoost Eğitim ve Doğrulama Kayıp Eğrisi')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('xgb_loss_curve_updated.png', dpi=300)
plt.show()

# Excel tablosuna dönüştür
excel_df = pd.DataFrame(excel_data)
excel_df = excel_df[['Model', 'Sınıf', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support']]
excel_df.to_excel('model_performance.xlsx', index=False)
print("\nModel performans sonuçları 'model_performance.xlsx' dosyasına kaydedildi.")