import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from nlp_preprocessing import IndonesianNLPPreprocessor

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def train_and_evaluate_models():
    print("=" * 70)
    print("  PELATIHAN MODEL NLP: TF-IDF + ML vs WORD EMBEDDING + DEEP LEARNING")
    print("  Studi Kasus: Agentic Campus - Academic Course Difficulty Advisor")
    print("=" * 70)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Pengumpulan Dataset
    dataset_path = "data/dataset_nlp_intent.csv"
    if not os.path.exists(dataset_path):
        from create_nlp_dataset import generate_agentic_campus_dataset
        generate_agentic_campus_dataset()
        
    df = pd.read_csv(dataset_path)
    print(f"\n[1] DATASET: Total {len(df)} sampel data query mahasiswa.")
    print("    Distribusi kelas intent:")
    for k, v in df["intent"].value_counts().items():
        print(f"     - {k:<28}: {v}")
        
    # 2. Pre-processing Data
    print("\n[2] PRE-PROCESSING DATA...")
    print("    Tahapan: Case Folding -> Cleansing -> Tokenizing -> Stopword Removal -> Stemming")
    preprocessor = IndonesianNLPPreprocessor(use_stemming=True)
    
    # Cek cache hasil preprocessing agar cepat
    preprocessed_csv = "data/dataset_nlp_intent_preprocessed.csv"
    if os.path.exists(preprocessed_csv):
        print(f"    Memuat cache preprocessed data dari {preprocessed_csv}...")
        df = pd.read_csv(preprocessed_csv)
    else:
        print("    Memproses teks kalimat satu per satu...")
        df["clean_text"] = df["text"].apply(lambda t: preprocessor.preprocess(t))
        df.to_csv(preprocessed_csv, index=False)
        print(f"    [OK] Preprocessing selesai & disimpan ke {preprocessed_csv}")
        
    # Tampilkan contoh hasil preprocessing
    print("\n    Contoh Perbandingan Teks Sebelum & Sesudah Preprocessing:")
    for i in range(min(3, len(df))):
        print(f"     * Raw   : {df['text'].iloc[i]}")
        print(f"       Clean : {df['clean_text'].iloc[i]}")
        
    # Encode Target Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["intent"])
    classes = label_encoder.classes_.tolist()
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    
    # Train-Test Split (80:20 Stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"], y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\n    Train set : {len(X_train_raw)} sampel")
    print(f"    Test set  : {len(X_test_raw)} sampel")
    
    # =========================================================================
    # 3. Representasi Terstruktur TF-IDF + Algoritma Machine Learning
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] REPRESENTASI TERSTRUKTUR: TF-IDF + ALGORITMA MACHINE LEARNING")
    print("=" * 70)
    
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
    joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.joblib")
    print(f"    TF-IDF Feature Matrix: {X_train_tfidf.shape[1]} fitur kosakata.")
    
    # Eksperimen Algoritma Machine Learning
    ml_models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(C=1.0, random_state=42)
    }
    
    ml_results = {}
    best_ml_name = ""
    best_ml_acc = -1
    best_ml_model = None
    
    for name, model in ml_models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        ml_results[name] = {"accuracy": float(acc), "f1_score": float(f1)}
        print(f"    * Model: {name:<24} | Akurasi: {acc * 100:.2f}% | F1-Score: {f1:.4f}")
        
        if acc > best_ml_acc:
            best_ml_acc = acc
            best_ml_name = name
            best_ml_model = model
            
    # Simpan model ML terbaik (Logistic Regression / Linear SVM)
    joblib.dump(best_ml_model, "models/ml_classifier.joblib")
    print(f"\n    [OK] Model ML Terbaik: {best_ml_name} ({best_ml_acc * 100:.2f}%) disimpan ke models/ml_classifier.joblib")
    
    # Classification Report Model ML
    y_pred_best_ml = best_ml_model.predict(X_test_tfidf)
    ml_report = classification_report(y_test, y_pred_best_ml, target_names=classes, output_dict=True)
    print("\n    Laporan Klasifikasi Model ML (" + best_ml_name + "):")
    print(classification_report(y_test, y_pred_best_ml, target_names=classes))
    
    # Plot Confusion Matrix ML
    cm_ml = confusion_matrix(y_test, y_pred_best_ml)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ml, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix: TF-IDF + {best_ml_name}")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_ml.png", dpi=150)
    plt.close()
    
    # =========================================================================
    # 4. Representasi Word Embedding + Algoritma Deep Learning
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4] REPRESENTASI WORD EMBEDDING + ALGORITMA DEEP LEARNING (BiLSTM)")
    print("=" * 70)
    
    vocab_size = 3000
    max_len = 35
    embedding_dim = 64
    
    # Tokenizer Keras
    dl_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    dl_tokenizer.fit_on_texts(X_train_raw)
    joblib.dump(dl_tokenizer, "models/dl_tokenizer.joblib")
    
    X_train_seq = dl_tokenizer.texts_to_sequences(X_train_raw)
    X_test_seq = dl_tokenizer.texts_to_sequences(X_test_raw)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")
    
    num_classes = len(classes)
    
    # Arsitektur Deep Learning: Embedding Layer + BiLSTM
    dl_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        SpatialDropout1D(0.25),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])
    
    dl_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("    Arsitektur Model Deep Learning:")
    dl_model.summary(print_fn=lambda x: print(f"    {x}"))
    
    # Training dengan EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    print("\n    Memulai Training Model Deep Learning (15 Epochs)...")
    history = dl_model.fit(
        X_train_pad, y_train,
        validation_data=(X_test_pad, y_test),
        epochs=15,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Simpan Model DL
    dl_model.save("models/dl_bilstm.keras")
    print("    [OK] Model Deep Learning disimpan ke models/dl_bilstm.keras")
    
    # Evaluasi Deep Learning
    test_loss, test_acc = dl_model.evaluate(X_test_pad, y_test, verbose=0)
    y_pred_probs = dl_model.predict(X_test_pad, verbose=0)
    y_pred_dl = np.argmax(y_pred_probs, axis=1)
    f1_dl = f1_score(y_test, y_pred_dl, average="weighted")
    
    print(f"\n    * Hasil Evaluasi Deep Learning (BiLSTM):")
    print(f"      - Akurasi Test : {test_acc * 100:.2f}%")
    print(f"      - Loss Test    : {test_loss:.4f}")
    print(f"      - F1-Score     : {f1_dl:.4f}")
    
    dl_report = classification_report(y_test, y_pred_dl, target_names=classes, output_dict=True)
    print("\n    Laporan Klasifikasi Model Deep Learning (Word Embedding + BiLSTM):")
    print(classification_report(y_test, y_pred_dl, target_names=classes))
    
    # Plot Confusion Matrix DL
    cm_dl = confusion_matrix(y_test, y_pred_dl)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_dl, annot=True, fmt="d", cmap="Purples",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix: Word Embedding + BiLSTM")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_dl.png", dpi=150)
    plt.close()
    
    # Plot Training History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", color="orange")
    plt.title("Deep Learning: Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Val Loss", color="orange")
    plt.title("Deep Learning: Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("reports/training_history_dl.png", dpi=150)
    plt.close()
    
    # =========================================================================
    # 5. Rangkuman & Perbandingan Hasil
    # =========================================================================
    comparison = {
        "models_evaluated": {
            "TF-IDF + " + best_ml_name: {
                "accuracy": round(float(best_ml_acc), 4),
                "f1_score": round(float(f1_score(y_test, y_pred_best_ml, average="weighted")), 4),
                "report": ml_report
            },
            "Word Embedding + BiLSTM": {
                "accuracy": round(float(test_acc), 4),
                "f1_score": round(float(f1_dl), 4),
                "loss": round(float(test_loss), 4),
                "report": dl_report
            }
        },
        "all_ml_algorithms": ml_results,
        "classes": classes
    }
    
    with open("models/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print("\n[OK] Seluruh metrik evaluasi dan laporan perbandingan disimpan ke:")
    print("     - models/evaluation_results.json")
    print("     - reports/confusion_matrix_ml.png")
    print("     - reports/confusion_matrix_dl.png")
    print("     - reports/training_history_dl.png")
    print("=" * 70)
    print("PELATIHAN SELESAI DENGAN SUKSES!")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate_models()
