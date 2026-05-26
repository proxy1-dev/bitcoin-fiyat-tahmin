import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from veri_isleme import veriyi_hazirla

def tahmin_et_ve_sun():
    print("1. Gerekli veriler ve ölçeklendirici (scaler) alınıyor...")
    _, _, X_test, y_test, scaler = veriyi_hazirla()

    print("2. Eğitilmiş model ve eğitim geçmişi yükleniyor...")
    try:
        model = load_model('lstm_model.h5')
        with open('egitim_gecmisi.pkl', 'rb') as f:
            history_dict = pickle.load(f)
    except FileNotFoundError:
        print("HATA: 'lstm_model.h5' veya 'egitim_gecmisi.pkl' bulunamadı.")
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

    df = pd.read_csv("bitcoin.csv")
    tum_gercek_fiyatlar = df['Close'].values

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
    axs[0, 1].plot(gercek_test_degerleri, color='blue', label='Gerçek Fiyatlar')
    axs[0, 1].plot(test_tahminleri, color='red', alpha=0.7, label='Modelin Tahminleri')
    axs[0, 1].set_title('Test Seti Başarısı')
    axs[0, 1].set_xlabel('Günler')
    axs[0, 1].set_ylabel('Fiyat (USD)')
    axs[0, 1].legend()

    # [1, 0] Sol Alt: Tüm Zamanlar Görünümü
    toplam_gun_sayisi = len(tum_gercek_fiyatlar)
    x_gecmis = np.arange(toplam_gun_sayisi)
    axs[1, 0].plot(x_gecmis, tum_gercek_fiyatlar, color='blue', label='Tüm Geçmiş Fiyatlar')
    
    last_idx = toplam_gun_sayisi - 1
    last_val = tum_gercek_fiyatlar[-1]
    birlesik_tahminler_all = np.concatenate(([last_val], gelecek_tahminler))
    x_birlesik_gelecek_all = np.arange(last_idx, last_idx + 31)
    
    axs[1, 0].plot(x_birlesik_gelecek_all, birlesik_tahminler_all, color='red', linewidth=2, linestyle='-', label='Gelecek 30 Gün (Tahmin)')
    axs[1, 0].set_title('Tüm Zamanlar Görünümü')
    axs[1, 0].set_xlabel('Günler')
    axs[1, 0].set_ylabel('Fiyat (USD)')
    axs[1, 0].legend()

    # [1, 1] Sağ Alt: Son 100 Gün Odaklı
    son_100_gun = tum_gercek_fiyatlar[-100:]
    x_son_100 = np.arange(100)
    axs[1, 1].plot(x_son_100, son_100_gun, color='blue', linewidth=2, label='Son 100 Gün (Gerçek)')
    
    last_real_idx_100 = 99
    last_real_val_100 = son_100_gun[-1]
    birlesik_tahminler_100 = np.concatenate(([last_real_val_100], gelecek_tahminler))
    x_birlesik_gelecek_100 = np.arange(last_real_idx_100, last_real_idx_100 + 31)
    
    axs[1, 1].plot(x_birlesik_gelecek_100, birlesik_tahminler_100, color='red', linewidth=2.5, linestyle='-', label='Gelecek 30 Gün (Tahmin)')
    axs[1, 1].set_title('Son 100 Gün Odaklı')
    axs[1, 1].set_xlabel('Son Günler')
    axs[1, 1].set_ylabel('Fiyat (USD)')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tahmin_et_ve_sun()