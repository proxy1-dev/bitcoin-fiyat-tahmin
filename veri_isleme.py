import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def veriyi_hazirla(dosya_yolu="bitcoin.csv", seq_length=60):

    print("1. Veri seti yukleniyor ve temizleniyor...")
    df = pd.read_csv(dosya_yolu)
    
    # Tarih formatını ayarlama ve veriyi tarihe göre sıralama
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Eksik (NaN) verileri bir önceki günün kapanış fiyatıyla doldurma
    df['Close'] = df['Close'].ffill()
    
    # Sadece Kapanış (Close) fiyatlarını alıp Numpy dizisine çevirme
    series = df['Close'].values.reshape(-1, 1)

    print("2. Veriler Egitim (%80) ve Test (%20) olarak bolunuyor...")
    split = int(len(series) * 0.8)
    train_data = series[:split]
    test_data = series[split:]

    print("3. Veriler 0-1 arasina olceklendiriliyor (Min-Max Scaling)...")
    # Ölçeklendiriciyi SADECE eğitim verisine fit ediyoruz
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    print(f"4. Veriler {seq_length} gunluk pencerelere (zaman adimlarina) bolunuyor...")
    X_train, y_train = create_sequences(scaled_train, seq_length)
    X_test, y_test = create_sequences(scaled_test, seq_length)

    # LSTM modelinin beklediği 3 boyutlu formata (Örnek Sayısı, Zaman Adımı, Özellik Sayısı) dönüştürme
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print("Veri hazirligi basariyla tamamlandi!\n")
    
    # Hazırlanan verileri ve 'scaler' objesini geri döndürüyoruz
    return X_train, y_train, X_test, y_test, scaler