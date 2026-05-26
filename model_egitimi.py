# model_egitimi.py
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Aşamadaki (arkadaşının yazdığı) veri hazırlama fonksiyonunu çağırıyoruz
from veri_isleme import veriyi_hazirla

def modeli_egit_ve_kaydet():
    print("1. Veriler işleme katmanından çekiliyor...")
    # Eğitim ve test verilerini alıyoruz
    X_train, y_train, X_test, y_test, scaler = veriyi_hazirla()

    print("2. LSTM Modeli inşa ediliyor...")
    model = Sequential()
    
    # 1. LSTM Katmanı ve Dropout (Overfitting engellemek için)
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    
    # 2. LSTM Katmanı
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Çıkış Katmanı (Tek fiyat tahmini)
    model.add(Dense(units=1))

    # Modeli derleme
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("3. Model eğitimi başlıyor (Bu işlem biraz vakit alabilir)...")
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("4. Model ve eğitim geçmişi kaydediliyor...")
    # Modeli kaydetme (h5 formatında)
    model.save('lstm_model.h5')
    print("-> Model başarıyla 'lstm_model.h5' olarak kaydedildi!")

    # Eğitim geçmişini (loss verilerini) kaydetme
    with open('egitim_gecmisi.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("-> Eğitim geçmişi 'egitim_gecmisi.pkl' olarak kaydedildi!\n")
    print("2. Aşama başarıyla tamamlandı. Artık 'sunum_ve_tahmin.py' dosyasını çalıştırabilirsiniz.")

if __name__ == "__main__":
    modeli_egit_ve_kaydet()