# 🤗 Transformers ile Metin Sınıflandırma (AG News)

Bu notebook, **Hugging Face Transformers** kütüphanesini kullanarak haber metinlerini kategorilere ayıran bir metin sınıflandırma modelinin nasıl eğitileceğini adım adım göstermektedir. Eğitim materyali olarak hazırlanmış olup, her kod bloğu detaylı açıklamalarla birlikte sunulmaktadır.

---

## 📋 Genel Bakış

| Özellik       | Detay                                             |
| ------------- | ------------------------------------------------- |
| **Model**     | `distilbert-base-uncased`                         |
| **Veri Seti** | AG News (4 kategori, 120.000 eğitim / 7.600 test) |
| **Görev**     | Çok Sınıflı Metin Sınıflandırma                   |
| **Donanım**   | Intel Arc GPU (XPU) destekli                      |
| **Doğruluk**  | ~%94 (test seti)                                  |

---

## 🗂️ İçindekiler

1. [Donanım Kontrolü](#1-donanım-kontrolü)
2. [Veri Setinin Yüklenmesi ve Keşfi](#2-veri-setinin-yüklenmesi-ve-keşfi)
3. [Tokenization (Tokenizasyon)](#3-tokenization)
4. [Veri Setinin Bölünmesi](#4-veri-setinin-bölünmesi)
5. [Model Yükleme ve Eğitim](#5-model-yükleme-ve-eğitim)
6. [Değerlendirme ve Sonuçlar](#6-değerlendirme-ve-sonuçlar)

---

## 🏷️ Kategoriler

AG News veri seti 4 haber kategorisi içerir:

| ID  | Kategori                      |
| --- | ----------------------------- |
| 0   | 🌍 World (Dünya)              |
| 1   | ⚽ Sports (Spor)              |
| 2   | 💼 Business (İş)              |
| 3   | 🔬 Sci/Tech (Bilim/Teknoloji) |

---

## 🔄 Pipeline Özeti

```
Ham Metin → Tokenizer → Token ID'leri → DistilBERT → Sınıf Tahmini
```

```
AG News Dataset
      ↓
AutoTokenizer (distilbert-base-uncased)
  - truncation=True
  - padding="max_length"
  - max_length=128
      ↓
AutoModelForSequenceClassification
  - num_labels=4
  - Transfer Learning
      ↓
Trainer API
  - 1 Epoch
  - batch_size=64
  - weight_decay=0.01
      ↓
~%94 Doğruluk
```

---

## 📦 Gereksinimler

```bash
pip install torch
pip install transformers
pip install datasets
pip install evaluate
pip install scikit-learn
pip install matplotlib seaborn
pip install pandas numpy
```

---

## 🚀 Hızlı Başlangıç

Notebook'u sırayla çalıştırın:

### 1. Donanım Kontrolü

Eğitim için kullanılabilir GPU/XPU varlığını kontrol eder.

```python
import torch
torch.xpu.is_available()  # Intel Arc GPU için
```

### 2. Veri Setinin Yüklenmesi ve Keşfi

```python
from datasets import load_dataset

dataset = load_dataset("ag_news")
# 120.000 eğitim, 7.600 test verisi
```

**Veri yapısı:**

- `text`: Ham haber metni
- `label`: Kategori ID'si (0-3)

Veri seti **dengeli** olup her kategoride eşit sayıda örnek bulunmaktadır (30.000 adet/kategori).

### 3. Tokenization

`AutoTokenizer`, metni modelin anlayabileceği sayısal ID dizilerine dönüştürür.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

**Tokenizasyon nasıl çalışır?**

- Kelimeler, modelin 30.522 kelimelik sözlüğündeki ID'lere eşlenir
- Bilinmeyen/uzun kelimeler alt parçalara bölünür: `dataset` → `['data', '##set']`
- `[CLS]` (101) ve `[SEP]` (102) özel token'ları eklenir
- Kısa metinler `padding` (0) ile, uzun metinler `truncation` ile 128 uzunluğa getirilir

**Tokenize edilmiş örnek:**

- `input_ids`: Gerçek kelime ID'leri
- `attention_mask`: Hangi token'ların gerçek (1), hangilerinin dolgu (0) olduğu
- `token_type_ids`: Cümle ayrımı için (tek cümle senaryosunda hepsi 0)

### 4. Veri Setinin Bölünmesi

```python
# Train'i %95 eğitim, %5 validation olarak böl
split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
# → 114.000 eğitim, 6.000 validation
# dataset["test"] → 7.600 (dokunulmadı, final test için)
```

### 5. Model Yükleme ve Eğitim

**Transfer Learning** sayesinde DistilBERT'in dil bilgisi korunur, sadece sınıflandırma kafası eklenir:

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

training_args = TrainingArguments(
    output_dir="./model_xx",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

**Eğitim İstatistikleri:**

- 1.782 güncelleme adımı
- ~36 dakika eğitim süresi
- ~45 örnek/saniye

### 6. Değerlendirme ve Sonuçlar

#### Test Seti Sonuçları

| Kategori     | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| World (0)    | 0.96      | 0.95   | 0.95     |
| Sports (1)   | 0.99      | 0.99   | 0.99     |
| Business (2) | 0.92      | 0.91   | 0.92     |
| Sci/Tech (3) | 0.90      | 0.93   | 0.92     |
| **Accuracy** |           |        | **0.94** |

#### Tek Metin Tahmini

```python
def tahmin_et(metin):
    inputs = tokenizer(metin, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    print(f"Tahmin: Sınıf {pred}, Güven: %{confidence*100:.1f}")

tahmin_et("Iran and america fighting each other for oil")
# → Tahmin: Sınıf 3 (Sci/Tech), Güven: %96.9
```

---

## 💡 Öğrenilen Kavramlar

| Kavram                   | Açıklama                                      |
| ------------------------ | --------------------------------------------- |
| **Transfer Learning**    | Önceden eğitilmiş modeli yeni göreve uyarlama |
| **Tokenization**         | Metni sayısal ID'lere dönüştürme              |
| **Padding & Truncation** | Sabit uzunlukta vektör oluşturma              |
| **Attention Mask**       | Gerçek token'ları dolgudan ayırt etme         |
| **Classification Head**  | Modele eklenen sınıflandırma katmanı          |
| **Warmup Steps**         | Öğrenme oranını yavaşça artırma               |
| **Weight Decay**         | Overfitting'i önleme tekniği                  |
| **Batched Map**          | Büyük veri setlerini bellek dostu işleme      |

---

## 🔗 Kaynaklar

- [Hugging Face Transformers Dokümantasyonu](https://huggingface.co/docs/transformers)
- [AG News Veri Seti](https://huggingface.co/datasets/ag_news)
- [DistilBERT Model Kartı](https://huggingface.co/distilbert-base-uncased)
- [Hugging Face Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)

---

## 📄 Lisans

Bu notebook eğitim amaçlı hazırlanmıştır ve serbestçe kullanılabilir.
