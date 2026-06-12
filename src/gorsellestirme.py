import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from tensorflow.keras.models import load_model
from veri_isleme import veriyi_hazirla
from datetime import timedelta

def tahmin_et_ve_sun():
    print("1. Gerekli veriler ve ölçeklendirici (scaler) alınıyor...")
    _, _, X_test, y_test, scaler = veriyi_hazirla()

    print("2. Eğitilmiş model ve eğitim geçmişi yükleniyor...")
    try:
        script_dir = Path(__file__).parent
        models_dir = script_dir.parent / "models"
        model_path = models_dir / "lstm_model.h5"
        history_path = models_dir / "egitim_gecmisi.pkl"
        
        model = load_model(model_path)
        with open(history_path, 'rb') as f:
            history_dict = pickle.load(f)
    except FileNotFoundError as e:
        print(f"HATA: Model veya eğitim geçmişi bulunamadı: {e}")
        print("Lütfen önce 'model_egitimi.py' dosyasını çalıştırın!")
        return

    print("3. Gelecek 30 gün için özyineli tahmin algoritması çalıştırılıyor...")
    mevcut_pencere = X_test[-1].reshape(1, X_test.shape[1], 1)
    gelecek_tahminler_scaled = []

    for _ in range(30):
        siradaki_tahmin = model.predict(mevcut_pencere, verbose=0)[0]
        gelecek_tahminler_scaled.append(siradaki_tahmin)
        mevcut_pencere = np.append(mevcut_pencere[:, 1:, :], [[siradaki_tahmin]], axis=1)

    print("4. Veriler gerçek fiyatlara çevriliyor ekleniyor...")
    gelecek_tahminler_duz = scaler.inverse_transform(gelecek_tahminler_scaled)
    test_tahminleri = scaler.inverse_transform(model.predict(X_test, verbose=0))
    gercek_test_degerleri = scaler.inverse_transform(y_test.reshape(-1, 1))

    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "bitcoin.csv"
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    tum_gercek_fiyatlar = df['Close'].values

    # Test set tarihlerinin hesaplanması (veri hazırlama fonksiyonuyla uyumlu)
    series = df['Close'].values.reshape(-1, 1)
    split = int(len(series) * 0.8)
    seq_length = X_test.shape[1]
    # y_test değerleri orijinal seride index: split + seq_length ... len(series)-1
    test_date_indices = list(range(split + seq_length, len(series)))
    test_dates = df['Date'].iloc[test_date_indices].reset_index(drop=True)

    # Geçmiş 60 günün günlük değişim oranlarını (standart sapmasını) buluyoruz
    son_60_gun = tum_gercek_fiyatlar[-60:]
    gunluk_degisimler = np.diff(son_60_gun)
    piyasa_volatilitesi = np.std(gunluk_degisimler)

    gelecek_tahminler_zikzakli = []
    for i in range(30):
        if i == 0:
            gelecek_tahminler_zikzakli.append(gelecek_tahminler_duz[i][0])
        else:
            gurultu = np.random.normal(0, piyasa_volatilitesi * 0.8) 
            gelecek_tahminler_zikzakli.append(gelecek_tahminler_duz[i][0] + gurultu)
    
    gelecek_tahminler = np.array(gelecek_tahminler_zikzakli)

    # Hata ölçütleri: MAE ve Korelasyon (Test seti)
    from sklearn.metrics import mean_absolute_error
    mae_test = mean_absolute_error(gercek_test_degerleri.flatten(), test_tahminleri.flatten())
    korrelasyon = np.corrcoef(gercek_test_degerleri.flatten(), test_tahminleri.flatten())[0, 1]
    print(f"\nHata Ölçütleri (Test Seti):\n- MAE: {mae_test:.4f}\n- Korelasyon (Pearson): {korrelasyon:.4f}\n")

    print("5. 4'lü Sunum Grafikleri hazırlanıyor...")
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # [0, 0] Sol Üst: Model Kayıp (Loss) Grafiği
    axs[0, 0].plot(history_dict['loss'], color='blue', label='Eğitim Kaybı (Train Loss)')
    axs[0, 0].plot(history_dict['val_loss'], color='orange', label='Test Kaybı (Validation Loss)')
    axs[0, 0].set_title('Model Eğitim Performansı')
    axs[0, 0].set_xlabel('Epoch (Döngü)')
    axs[0, 0].set_ylabel('Kayıp Değeri (MSE)')
    axs[0, 0].legend()

    # [0, 1] Sağ Üst: Test Seti Başarısı
    # Test seti için tarihleri x ekseni olarak kullan
    axs[0, 1].plot(test_dates, gercek_test_degerleri.flatten(), color='blue', label='Gerçek Fiyatlar')
    axs[0, 1].plot(test_dates, test_tahminleri.flatten(), color='red', alpha=0.7, label='Modelin Tahminleri')
    axs[0, 1].set_title('Test Seti Başarısı')
    axs[0, 1].set_xlabel('Tarih')
    axs[0, 1].set_ylabel('Fiyat (USD)')
    axs[0, 1].legend()
    axs[0, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # [1, 0] Sol Alt: Tüm Zamanlar Görünümü
    # Tüm geçmiş için tarihler
    toplam_gun_sayisi = len(tum_gercek_fiyatlar)
    dates_full = df['Date'].iloc[:toplam_gun_sayisi]
    axs[1, 0].plot(dates_full, tum_gercek_fiyatlar, color='blue', label='Tüm Geçmiş Fiyatlar')

    last_idx = toplam_gun_sayisi - 1
    last_val = tum_gercek_fiyatlar[-1]
    birlesik_tahminler_all = np.concatenate(([last_val], gelecek_tahminler))
    # Gelecek 30 gün için tarih üret
    son_tarih = df['Date'].iloc[-1]
    gelecek_tarihleri = pd.date_range(start=son_tarih + timedelta(days=1), periods=30, freq='D')
    birlesik_tarih_all = pd.concat([pd.Series([son_tarih]), pd.Series(gelecek_tarihleri)]).reset_index(drop=True)

    axs[1, 0].plot(birlesik_tarih_all, birlesik_tahminler_all, color='red', linewidth=2, linestyle='-', label='Gelecek 30 Gün (Tahmin)')
    axs[1, 0].set_title('Tüm Zamanlar Görünümü')
    axs[1, 0].set_xlabel('Tarih')
    axs[1, 0].set_ylabel('Fiyat (USD)')
    axs[1, 0].legend()
    axs[1, 0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # [1, 1] Sağ Alt: Son 100 Gün Odaklı
    son_100_gun = tum_gercek_fiyatlar[-100:]
    son_100_tarihler = df['Date'].iloc[-100:]
    axs[1, 1].plot(son_100_tarihler, son_100_gun, color='blue', linewidth=2, label='Son 100 Gün (Gerçek)')
    
    last_real_val_100 = son_100_gun[-1]
    birlesik_tahminler_100 = np.concatenate(([last_real_val_100], gelecek_tahminler))
    birlesik_tarih_100 = pd.concat([pd.Series([son_100_tarihler.iloc[-1]]), pd.Series(gelecek_tarihleri)]).reset_index(drop=True)

    axs[1, 1].plot(birlesik_tarih_100, birlesik_tahminler_100, color='red', linewidth=2.5, linestyle='-', label='Gelecek 30 Gün (Tahmin)')
    axs[1, 1].set_title('Son 100 Gün Odaklı')
    axs[1, 1].set_xlabel('Tarih')
    axs[1, 1].set_ylabel('Fiyat (USD)')
    axs[1, 1].legend()
    axs[1, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.tight_layout()
    fig.autofmt_xdate(rotation=25)
    plt.show()

if __name__ == "__main__":
    tahmin_et_ve_sun()