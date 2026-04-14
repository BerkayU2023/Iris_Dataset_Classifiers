# İris Çiçeği Sınıflandırma - Uçtan Uca Makine Öğrenmesi Boru Hattı

Bu proje, makine öğrenmesi dünyasının en popüler veri setlerinden biri olan İris (Iris) veri setini kullanarak uçtan uca bir sınıflandırma boru hattı (pipeline) oluşturmayı amaçlamaktadır. Proje kapsamında veri keşfi (EDA), veri ön işleme, model eğitimi, hiperparametre optimizasyonu ve model değerlendirme adımları uygulanmıştır.

## Kullanılan Teknolojiler ve Kütüphaneler

Projenin çalışması için aşağıdaki Python kütüphaneleri kullanılmıştır:

* **Python 3.x**
* `pandas` (Veri manipülasyonu ve analizi)
* `numpy` (Sayısal hesaplamalar)
* `seaborn` & `matplotlib` (Veri görselleştirme ve ısı haritaları)
* `scikit-learn` (Makine öğrenmesi modelleri, ön işleme ve metrikler)

## Model Performansları ve Sonuçlar

Veri seti üzerinde eğitilen temel modellerin ve hiperparametre optimizasyonu (Tuning) yapılmış modellerin test seti (`test_size=0.25`) üzerindeki performansları aşağıdaki gibidir:

| Model | Test Doğruluğu (Accuracy) | Çapraz Doğrulama (CV) Skoru | En İyi Parametreler |
| :--- | :--- | :--- | :--- |
| **Gaussian Naive Bayes** | %100 (1.00) | - | *Varsayılan* |
| **Lojistik Regresyon (Temel)** | %100 (1.00) | - | *Varsayılan* |
| **Lojistik Regresyon (RandomizedSearchCV)** | %100 (1.00) | %95.5 | `{"solver": "saga", "penalty": "l1", "C": 1}` |
| **Lojistik Regresyon (GridSearchCV)** | %100 (1.00) | %95.5 | `{"solver": "sag", "penalty": "l2", "C": 1000}` |
| **Destek Vektör Makineleri (SVC - Tuned)** | %97.4 (0.97) | %95.5 | `{"kernel": "linear", "gamma": 1, "C": 10}` |

> **Not:** İris veri setinin basit yapısından dolayı temel modeller test setinde 1.0 skor üretmiştir. Ancak SVC üzerinde yapılan GridSearchCV optimizasyonu, çapraz doğrulama skorlarıyla tutarlı olarak daha gerçekçi (%97.4) bir genelleme performansı sergilemiş ve tek bir örneği (Sınıf 1'i Sınıf 2 olarak) yanlış sınıflandırarak aşırı öğrenmenin önüne geçmiştir.

## Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1.  Depoyu klonlayın:
    ```bash
    git clone [https://github.com/kullanici_adiniz/iris-ml-pipeline.git](https://github.com/kullanici_adiniz/iris-ml-pipeline.git)
    cd iris-ml-pipeline
    ```

2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```

3.  Ana Python dosyasını çalıştırın:
    ```bash
    python main.py
    ```

## Proje Adımları

1.  **Keşifsel Veri Analizi (EDA):** Veri seti projeye dahil edilmiş, eksik değer kontrolü yapılmış ve özelliklerin sınıflara göre dağılımı `seaborn` saçılım grafikleri (scatterplot) ve `pairplot` ile görselleştirilmiştir.
2.  **Veri Ön İşleme:**
    * Kategorik hedef değişken (`Species`), `LabelEncoder` kullanılarak sayısal değerlere dönüştürülmüştür.
    * "Id" sütunu veri setinden çıkarılmıştır.
    * Veri seti eğitim (%75) ve test (%25) setlerine ayrılmıştır.
    * Mesafe tabanlı algoritmaların doğru çalışabilmesi için özellikler `StandardScaler` ile ölçeklendirilmiştir (Veri sızıntısını önlemek için ölçeklendirici sadece eğitim setine `fit` edilmiştir).
3.  **Model Eğitimi ve Optimizasyon:** Lojistik Regresyon ve SVC için uyumlu parametre ızgaraları (param_grid) kurularak, Çapraz Doğrulama (`StratifiedKFold`) ve `GridSearchCV` ile en iyi hiperparametreler tespit edilmiştir.
4.  **Değerlendirme:** Modellerin tahmin gücü; Doğruluk Skoru (Accuracy), Sınıflandırma Raporu (Classification Report) ve Karmaşıklık Matrisi (Confusion Matrix) Isı Haritaları çizdirilerek incelenmiştir.
