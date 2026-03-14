# Keras 3 ile Deep Learning - California Housing Projesi

Bu proje, **California Housing** veri setini kullanarak iki farklı Jupyter Notebook içermektedir:

1. **Keşifsel Veri Analizi (EDA)** — `Housing_Analiz_.ipynb`
2. **Derin Öğrenme Modeli (Keras 3 + JAX)** — `keras3_jax_ile_deep_learning.ipynb`

---

## 📁 Dosyalar

### 1. `Housing_Analiz_.ipynb` — Veri Analizi

California Housing veri setini derinlemesine inceleyen bu notebook aşağıdaki adımları kapsar:

| Adım                 | Açıklama                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------- |
| Veri Yükleme         | `sklearn.datasets.fetch_california_housing` ile veri yüklenir                             |
| Keşifsel Analiz      | `describe()`, `info()`, `head()` ile genel istatistikler                                  |
| Sütun İsimlendirme   | Sütun adları Türkçeye çevrilir                                                            |
| Aykırı Değer Analizi | IQR yöntemiyle her sütun için alt/üst sınırlar ve aykırı sayıları hesaplanır              |
| Görselleştirme       | Histogram grafikleri ile dağılım analizi                                                  |
| Aykırı Değer İşleme  | Aykırı değerler NaN olarak işaretlenir; enlem grubuna göre medyan/ortalama ile doldurulur |

#### Veri Seti Özellikleri

| Özellik       | Açıklama                                           |
| ------------- | -------------------------------------------------- |
| `MedInc`      | Medyan gelir (10.000$ birimi)                      |
| `HouseAge`    | Medyan konut yaşı                                  |
| `AveRooms`    | Hane başına ortalama oda sayısı                    |
| `AveBedrms`   | Hane başına ortalama yatak odası sayısı            |
| `Population`  | Bölge nüfusu                                       |
| `AveOccup`    | Hanedeki ortalama kişi sayısı                      |
| `Latitude`    | Enlem                                              |
| `Longitude`   | Boylam                                             |
| `MedHouseVal` | Medyan ev değeri (hedef değişken, 100.000$ birimi) |

---

### 2. `keras3_jax_ile_deep_learning.ipynb` — Derin Öğrenme Modeli

Keras 3 ve JAX backend'i kullanarak California Housing veri seti üzerinde bir regresyon modeli eğiten notebook:

#### Kullanılan Teknolojiler

- **Keras 3** — Yüksek seviyeli derin öğrenme API'si
- **JAX** — Hızlı eğitim için Google'ın hesaplama kütüphanesi
- **scikit-learn** — Veri ön işleme ve metrik hesaplama

#### Model Mimarisi

```
Input(8 özellik)
  → Dense(128, ReLU)
  → Dense(64, ReLU)
  → Dense(32, ReLU)
  → Dense(1)  # Çıkış katmanı (fiyat tahmini)
```

#### Eğitim Süreci

| Parametre        | Değer                     |
| ---------------- | ------------------------- |
| Optimizer        | Adam                      |
| Loss             | MSE (Mean Squared Error)  |
| Metrik           | MAE (Mean Absolute Error) |
| Epoch            | 50                        |
| Batch Size       | 32                        |
| Validation Split | %10                       |

#### Test Sonuçları

| Metrik   | Değer  |
| -------- | ------ |
| MAE      | ~0.335 |
| R² Skoru | ~0.802 |

> R² = 0.80 demek, modelin ev fiyatlarındaki değişimin **%80'ini** doğru tahmin edebildiği anlamına gelir.

---

## 🚀 Kurulum ve Çalıştırma

### Gerekli Kütüphaneler

```bash
pip install keras
pip install jax jaxlib
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Çalıştırma

```bash
jupyter notebook
```

Ardından `Housing_Analiz_.ipynb` veya `keras3_jax_ile_deep_learning.ipynb` dosyasını açın.

---

## 📊 Veri Seti Hakkında

California Housing veri seti, 1990 ABD nüfus sayımından türetilmiştir. Her satır bir nüfus sayım bloğu grubunu temsil eder. Toplam **20.640** örnek ve **8 özellik** içerir.

Kaynak: [StatLib Repository](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
