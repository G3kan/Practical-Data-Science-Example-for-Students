Veri Analizi ve Yapay Zeka ile Veri Tamamlama
Bu proje, e-ticaret müşteri yorumlarını analiz ederek "kirli" veriyi temizlemeyi, eksik verileri yerel bir yapay zeka (Ollama) ile doldurmayı ve iş zekasına (BI) hazır hale getirmeyi kapsar.

Uygulanan Adımlar
Veri Keşfi: 23.486 satırlık veri seti yüklendi, sütun isimleri Türkçeleştirildi ve istatistiksel özetler çıkarıldı.

Görsel Analiz: Müşteri yaş dağılımı, memnuniyet oranları ve departman bazlı puanlar Seaborn/Matplotlib ile görselleştirildi.

Eksik Veri Yönetimi: Isı haritası (Heatmap) ile eksik veriler tespit edildi.

Yapay Zeka (Ollama) Entegrasyonu: Başlığı boş olan yorumlar için qwen2.5:7b modeli kullanılarak otomatik başlıklar üretildi.

Segmentasyon: Müşteriler yaş gruplarına ayrıldı ve "Mutsuz Müşteri Oranı" gibi kritik metrikler hesaplandı.

Çıktı: Analiz edilen ve temizlenen veri Sonhali.csv olarak dışa aktarıldı.

Önemli Bulgular
Eksik verilerin yoğunlaştığı alanlar görsel olarak belirlendi.

Departman bazlı şikayet oranları hesaplanarak operasyonel riskler saptandı.

Yerel LLM kullanımı ile manuel veri giriş yükü azaltıldı.
