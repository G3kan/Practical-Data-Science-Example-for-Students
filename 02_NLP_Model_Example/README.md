Bu çalışma, ham metin verilerini kullanarak bir makine öğrenmesi modeli (Lojistik Regresyon) eğitmeyi ve bu modeli gelecekteki tahminler için kaydetmeyi amaçlar.

Proje Akışı
Veri Hazırlığı: Metinler temizlenir ve eğitim/test setlerine ayrılır.

Vektörleştirme (TF-IDF): Kelimeler, bilgisayarın anlayabileceği sayısal değerlere dönüştürülür.

Model Eğitimi: Lojistik Regresyon algoritması ile sınıflandırma modeli oluşturulur.

Performans Ölçümü: Doğruluk (Accuracy) ve Karmaşıklık Matrisi (Confusion Matrix) ile başarı analiz edilir.

Model Kaydı: Eğitilen model joblib ile saklanarak tekrar kullanıma hazır hale getirilir.

Temel Çıktılar
Sınıflandırma Raporu (Hassasiyet, Geri Çağırma, F1-Skoru).

Hata Matrisi Görselleştirmesi (Seaborn/Matplotlib).
