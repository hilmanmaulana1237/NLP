import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class IndonesianNLPPreprocessor:
    """
    Pipeline Preprocessing Teks Bahasa Indonesia untuk Tugas NLP Agentic Campus:
    1. Case Folding (mengubah huruf kecil)
    2. Cleansing (membersihkan tanda baca, angka, simbol)
    3. Tokenization (pemecahan kata/token)
    4. Stopword Removal (penghapusan kata umum menggunakan Sastrawi)
    5. Stemming (pengembalian ke kata dasar menggunakan Sastrawi)
    """
    def __init__(self, use_stemming=True, custom_stopwords=None):
        self.use_stemming = use_stemming
        
        # Inisialisasi Stopword Remover Sastrawi
        stopword_factory = StopWordRemoverFactory()
        default_stopwords = set(stopword_factory.get_stop_words())
        
        # Tambahan stopwords umum percakapan akademis/mahasiswa
        additional_stopwords = {
            "yg", "yang", "dgn", "dengan", "nya", "nih", "dong", "sih", "deh", 
            "ya", "kan", "tuh", "kah", "kok", "min", "kak", "bang", "bot",
            "kalo", "kalau", "gimana", "bagaimana", "kenapa", "mengapa",
            "apa", "apakah", "aja", "saja", "ada", "bisa", "tolong", "mau",
            "ingin", "buat", "untuk", "pada", "di", "ke", "dari", "ini", "itu"
        }
        
        self.stopwords = default_stopwords.union(additional_stopwords)
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Inisialisasi Stemmer Sastrawi jika diaktifkan
        if self.use_stemming:
            stemmer_factory = StemmerFactory()
            self.stemmer = stemmer_factory.create_stemmer()
        else:
            self.stemmer = None

    def case_folding(self, text: str) -> str:
        """Mengubah teks menjadi huruf kecil (lowercase)."""
        if not isinstance(text, str):
            text = str(text)
        return text.lower()

    def cleansing(self, text: str) -> str:
        """
        Membersihkan karakter tanda baca, URL, angka yang tidak relevan,
        dan normalisasi spasi.
        """
        # Hapus URL
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        # Hapus email / mention / hashtag
        text = re.sub(r"[@#]\w+", " ", text)
        # Hapus tanda baca
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Hapus karakter non-alfanumerik/simbol
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        # Normalisasi spasi berlebih
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list:
        """Memecah teks menjadi daftar token kata."""
        return text.split()

    def remove_stopwords(self, tokens: list) -> list:
        """Menghapus kata-kata tidak bermakna (stopwords)."""
        return [word for word in tokens if word not in self.stopwords and len(word) > 1]

    def stem_tokens(self, tokens: list) -> list:
        """Mengembalikan token ke bentuk dasar (root word)."""
        if not self.stemmer:
            return tokens
        text = " ".join(tokens)
        stemmed_text = self.stemmer.stem(text)
        return stemmed_text.split()

    def preprocess(self, text: str, return_tokens: bool = False):
        """
        Menjalankan seluruh alur preprocessing data:
        Case Folding -> Cleansing -> Tokenizing -> Stopword Removal -> Stemming
        """
        # 1. Case Folding
        cf = self.case_folding(text)
        # 2. Cleansing
        cl = self.cleansing(cf)
        # 3. Tokenize
        tokens = self.tokenize(cl)
        # 4. Stopword Removal
        filtered_tokens = self.remove_stopwords(tokens)
        # 5. Stemming (opsional tapi sesuai silabus NLP)
        if self.use_stemming and self.stemmer:
            final_tokens = self.stem_tokens(filtered_tokens)
        else:
            final_tokens = filtered_tokens
            
        if return_tokens:
            return final_tokens
        return " ".join(final_tokens)

    def get_preprocessing_steps(self, text: str) -> dict:
        """
        Mengembalikan setiap tahapan langkah preprocessing untuk keperluan
        dokumentasi dan laporan praktikum NLP.
        """
        cf = self.case_folding(text)
        cl = self.cleansing(cf)
        tokens = self.tokenize(cl)
        filtered = self.remove_stopwords(tokens)
        stemmed = self.stem_tokens(filtered) if self.stemmer else filtered
        
        return {
            "original": text,
            "case_folding": cf,
            "cleansing": cl,
            "tokenizing": tokens,
            "stopword_removed": filtered,
            "stemming": stemmed,
            "final_text": " ".join(stemmed)
        }

if __name__ == "__main__":
    preprocessor = IndonesianNLPPreprocessor(use_stemming=True)
    sample = "Mata kuliah apa yang paling SUSAH di Jurusan TEKNIK INFORMATIKA? Tolong kasih tips belajarnya dong!!"
    steps = preprocessor.get_preprocessing_steps(sample)
    print("Demo Preprocessing NLP:")
    for k, v in steps.items():
        print(f" - {k.upper():<18}: {v}")
