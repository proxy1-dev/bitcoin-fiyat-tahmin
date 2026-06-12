# model_egitimi.py
import pickle
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from veri_isleme import veriyi_hazirla

def modeli_egit_ve_kaydet():
    print("1. Veriler işleme katmanından çekiliyor...")
    # Eğitim ve test verilerini al
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
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)  # Klasör yoksa oluştur
    
    model_path = models_dir / "lstm_model.h5"
    history_path = models_dir / "egitim_gecmisi.pkl"
    
    model.save(model_path)
    print(f"-> Model başarıyla '{model_path}' olarak kaydedildi!")

    # Eğitim geçmişini (loss verilerini) kaydetme
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"-> Eğitim geçmişi '{history_path}' olarak kaydedildi!\n")
    print("2. Aşama başarıyla tamamlandı. Artık 'sunum_ve_tahmin.py' dosyasını çalıştırabilirsiniz.")

if __name__ == "__main__":
    modeli_egit_ve_kaydet()