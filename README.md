# 📈 LSTM Ağları ile Bitcoin (BTC) Fiyat Tahmini ve 30 Günlük Gelecek Projeksiyonu

Bu proje, tarihsel Bitcoin fiyat hareketlerini analiz ederek derin öğrenme tabanlı **Uzun-Kısa Süreli Bellek (LSTM)** mimarisiyle geleceğe yönelik özyineli (recursive) fiyat tahminleri gerçekleştiren uçtan uca bir veri bilimi projesidir.

---

## 📂 1. Klasör Yapısı (Repository Structure)

Deponun temiz, modüler ve kurumsal standartlara uygun görünmesi için tasarım şu şekildedir:

```text
bitcoin-price-prediction-lstm/
│
├── data/
│   └── bitcoin.csv              # Ham zaman serisi veri seti
│
├── src/
│   ├── veri_isleme.py           # Veri temizleme, ölçekleme ve pencereleme
│   ├── model_egitimi.py         # LSTM sinir ağı mimarisi ve eğitim döngüsü
│   └── gorsellestirme.py        # 30 günlük özyineli tahmin ve grafik motoru
│
├── models/                      # .gitignore ile takip edilmeyen, yerel model klasörü
│   ├── lstm_model.h5            # Eğitilmiş TensorFlow/Keras model dosyası (Yerel)
│   └── egitim_gecmisi.pkl       # Eğitim loss değerlerini barındıran geçmiş dosyası
│
├── .gitignore                   # GitHub'a yüklenmeyecek dosyalar listesi
├── requirements.txt             # Bağımlı olunan kütüphaneler listesi
└── README.md                    # Proje ana tanıtım ve kullanım kılavuzu
```

## 💻 Donanım ve Yazılım Gereksinimleri
Donanım:

İşlemci (CPU): Minimum 4 çekirdekli modern bir işlemci.

Bellek (RAM): Minimum 8 GB RAM (Eğitim sırasındaki veri yükleme işlemleri için).

Ekran Kartı (GPU): Model eğitimi için NVIDIA GPU önerilir (CUDA destekli). TensorFlow GPU hızlandırması kullanarak iterasyon (epoch) sürelerini önemli ölçüde kısaltabilirsiniz.

Yazılım:

İşletim Sistemi: Windows, Linux veya macOS

Python Sürümü: Python 3.8 veya üzeri

Sürüm Kontrolü: Git ve GitHub Desktop

## 📦 Kullanılan Kütüphaneler ve Açıklamaları

Proje kapsamında kullanılan temel kütüphaneler ve projedeki görevleri şu şekildedir:

| Kütüphane | Projedeki Görevi / Amacı |
| :--- | :--- |
| **TensorFlow / Keras** | Çok katmanlı LSTM yapısının kurulması, Dropout optimizasyonu ve modelin eğitilmesi. |
| **Pandas** | CSV formatındaki zaman serisi verilerinin okunması, `ffill` ile eksik verilerin doldurulması ve tarih indeksleme. |
| **NumPy** | Verilerin matris formuna getirilmesi, kaydırılan pencerelerin (sequences) dizi işlemlerinin yönetilmesi. |
| **Scikit-Learn** | Verilerin 0-1 arasına sıkıştırılması için `MinMaxScaler` ölçekleyicisinin sağlanması. |
| **Matplotlib** | Eğitim geçmişi (loss), geçmiş test verisi başarısı ve 30 günlük gelecek tahmin grafiğinin çizilmesi. |

## 📊 4. Veri Seti Bilgisi ve Örnek Veri

### Veri Kaynağı
Projede kullanılan veri seti, küresel finans platformu **Yahoo Finance API** üzerinden günlük periyotlarla çekilmiş tarihsel Bitcoin verilerini içerir. 
> 🔗 [Veri Setinin Alındığı Finans Kaynağı (Yahoo Finance)](https://finance.yahoo.com/quote/BTC-USD/history/)

### Örnek Veri Yapısı (`bitcoin.csv`)
Veri seti zaman serisi analizine uygun olarak kronolojik sıralıdır. İlk 3 satırın yapısal görünümü:

| Date | Close | High | Low | Open | Volume |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2014-09-17 | 457.33 | 468.17 | 452.42 | 465.86 | 21,056,800 |
| 2014-09-18 | 424.44 | 456.85 | 413.10 | 456.86 | 34,483,200 |
| 2014-09-19 | 394.80 | 427.83 | 384.53 | 424.10 | 37,919,700 |

## 🤖 5. Eğitilmiş Model Dosyaları (`models/`)

Büyük boyutlu ikili (binary) dosyaların deponun boyutunu şişirmemesi için `lstm_model.h5` ve `egitim_gecmisi.pkl` dosyaları `.gitignore` dosyasına eklenerek GitHub dışında tutulmuştur. 

**Modeli kullanmak için iki seçenek mevcuttur:**
1. **Sıfırdan Eğitin:** `src/model_egitimi.py` dosyasını çalıştırarak modeli eğitebilir ve dosyaların otomatik olarak yerelde üretilmesini sağlayabilirsiniz.
2. **Yerel Klasöre Ekleyin:** Eğer elinizde hazır eğitilmiş model dosyaları varsa, ana dizinde `models/` adında bir klasör açıp bu iki dosyayı içine yerleştirerek doğrudan tahmin aşamasına geçebilirsiniz.

## ⚙️ 6. Programın Kurulumu ve Çalıştırılması

### 6.1 Gerekli Yazılımların Yüklenmesi

Projeyi çalıştırmadan önce, aşağıda verilen yazılımlar sisteminizde yüklü olmalıdır:

1. **Python (3.8 veya üzeri)**
   - Windows: [python.org](https://www.python.org/downloads/) adresinden indirip kurabilirsiniz
   - Linux/macOS: Terminalden `python3 --version` ile kontrol edebilirsiniz

2. **Git** (opsiyonel, fakat önerilir)
   - [git-scm.com](https://git-scm.com/) adresinden indirip kurabilirsiniz

### 6.2 Adım Adım Kurulum

#### Adım 1: Projeyi İndir
```bash
git clone https://github.com/proxy1-dev/bitcoin-fiyat-tahmin.git
cd bitcoin-fiyat-tahmin
```

Veya ZIP dosyasını indirdiyseniz, klasörü açıp içine girin.

#### Adım 2: Python Sanal Ortamı Oluştur (Önerilir)

Windows'ta:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS'ta:
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Not:** Sanal ortam oluşturduktan sonra terminalde `(venv)` ön eki görünür. Bu ortamdan çıkmak için `deactivate` komutunu çalıştırabilirsiniz.

#### Adım 3: Bağımlılıkları Yükle

Sanal ortam aktif iken, aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

Bu komut, `requirements.txt` dosyasında belirtilen tüm gerekli kütüphaneleri (TensorFlow, Pandas, NumPy, Scikit-Learn, Matplotlib) otomatik olarak yükleyecektir.

> **Not:** TensorFlow ilk yüklenmesinde biraz zaman alabilir. Lütfen işlemin tamamlanmasını bekleyin.

### 6.3 Programı Çalıştırma

Kurulum tamamlandıktan sonra, aşağıdaki adımları takip ederek programı çalıştırabilirsiniz:

#### **1. Adım: Verileri İşle ve Modeli Eğit**

```bash
python src/model_egitimi.py
```

Bu komut:
- `data/bitcoin.csv` dosyasındaki verileri okur
- Verileri temizler ve ölçekler
- LSTM modelini eğitir (25 epoch)
- Eğitilmiş modeli `models/lstm_model.h5` olarak kaydeder
- Eğitim geçmişini `models/egitim_gecmisi.pkl` olarak kaydeder

**Beklenen çıktı:**
```
1. Veriler işleme katmanından çekiliyor...
1. Veri seti yukleniyor ve temizleniyor...
2. Veriler Egitim (%80) ve Test (%20) olarak bolunuyor...
3. Veriler 0-1 arasina olceklendiriliyor (Min-Max Scaling)...
4. Veriler 60 gunluk pencerelere (zaman adimlarina) bolunuyor...
Veri hazirligi basariyla tamamlandi!

2. LSTM Modeli inşa ediliyor...
3. Model eğitimi başlıyor (Bu işlem biraz vakit alabilir)...
Epoch 1/25
...
Epoch 25/25
4. Model ve eğitim geçmişi kaydediliyor...
-> Model başarıyla 'lstm_model.h5' olarak kaydedildi!
-> Eğitim geçmişi 'egitim_gecmisi.pkl' olarak kaydedildi!
```

#### **2. Adım: Tahminleri Görselleştir**

```bash
python src/gorsellestirme.py
```

Bu komut:
- Eğitilmiş modeli yükler
- Test verileri üzerinde tahminler yapar
- Gelecek 30 gün için özyineli tahminler gerçekleştirir
- Eğitim loss grafiğini gösterir
- Geçmiş ve gelecek tahminlerinin karşılaştırma grafiğini çıkarır

**Beklenen çıktı:**
```
1. Gerekli veriler ve ölçeklendirici (scaler) alınıyor...
2. Eğitilmiş model ve eğitim geçmişi yükleniyor...
3. Gelecek 30 gün için özyineli tahmin algoritması çalıştırılıyor...
4. Veriler gerçek fiyatlara çevriliyor ekleniyor...
[Matplotlib windows açılarak grafikler gösterilir]
```

### 6.4 Hızlı Başlangıç (Komut Adımları)

Eğer modeliniz daha önceden eğitilmiş ise ve sadece tahminleri görmek istiyorsanız:

```bash
# 1. Sanal ortamı aktif et
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 2. Tahminleri görselleştir (model zaten mevcut ise)
python src/gorsellestirme.py
```

### 6.5 Sorun Giderme

| Sorun | Çözüm |
| :--- | :--- |
| **ModuleNotFoundError: No module named 'tensorflow'** | `pip install -r requirements.txt` komutu ile kütüphaneleri yeniden yükleyin |
| **FileNotFoundError: 'lstm_model.h5' bulunamadı** | `python src/model_egitimi.py` ile modeli önce eğitin |
| **OutOfMemory (OOM) hatası** | RAM sınırlaması varsa, `model_egitimi.py` dosyasındaki `batch_size` değerini küçültmeyi deneyin (örn: 16) |
| **GPU hızlandırması çalışmıyor** | TensorFlow GPU desteği için NVIDIA CUDA Toolkit ve cuDNN yüklü olması gerekir |

---

