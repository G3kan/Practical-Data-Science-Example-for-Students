# MLflow ile Model Deneyimi Takibi

MLflow kütüphanesi kullanarak makine öğrenmesi deneylerinin parametrelerini, metriklerini ve modellerini takip etme örnekleri.

## İçerik

- Mlfow_example.ipynb: MLflow temel kullanımı ve deneyimi takibi

## Gerekli Kütüphaneler

```bash
pip install mlflow scikit-learn pandas numpy
```

## MLflow Temel Bileşenleri

1. Tracking: Parametreleri ve metrikleri kaydetme
2. Projects: Projeleri paketleme ve çalıştırma
3. Models: Model versiyonlama
4. Registry: Modellerini yönetme

## Kullanım

```python
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("accuracy", 0.95)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
```

MLflow UI'ye erişmek için `mlflow ui` komutunu çalıştırın.
