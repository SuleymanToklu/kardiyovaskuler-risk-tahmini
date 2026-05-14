"""
database_setup.py
-----------------
Makale reprodüksiyonu projesi — Adım 1: Veritabanı kurulumu ve ham veri aktarımı.

Bu modül şu işlemleri gerçekleştirir:
  1. `cardio_train.csv` dosyasını Pandas ile okur.
  2. `heart_disease.db` adlı bir SQLite veritabanı oluşturur.
  3. Ham veriyi hiçbir değişiklik yapmadan `raw_data` tablosuna yazar.

Kullanım:
  python database_setup.py
"""

import sqlite3
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent          # Scriptin bulunduğu dizin
CSV_PATH = BASE_DIR / "cardio_train.csv"  # Ham veri dosyası
DB_PATH  = BASE_DIR / "heart_disease.db"  # Hedef SQLite veritabanı
TABLE_NAME = "raw_data"                   # Hedef tablo adı

# CSV okuma parametreleri (cardio_train.csv noktalı virgülle ayrılmış)
CSV_READ_PARAMS: dict = {
    "sep": ";",       # Alan ayracı
    "index_col": 0,   # İlk sütun (id) indeks olarak kullanılır
}

# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> pd.DataFrame:
    """CSV dosyasını Pandas DataFrame olarak yükler.

    Parameters
    ----------
    csv_path : Path
        Okunacak CSV dosyasının yolu.

    Returns
    -------
    pd.DataFrame
        Ham veri içeren DataFrame.

    Raises
    ------
    FileNotFoundError
        Dosya bulunamazsa fırlatılır.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")

    logger.info("CSV okunuyor: %s", csv_path)
    df = pd.read_csv(csv_path, **CSV_READ_PARAMS)
    logger.info(
        "CSV başarıyla okundu → %d satır, %d sütun. Sütunlar: %s",
        len(df),
        len(df.columns),
        df.columns.tolist(),
    )
    return df


def save_to_db(
    df: pd.DataFrame,
    db_path: Path,
    table: str,
    if_exists: str = "replace",
) -> None:
    """DataFrame'i SQLite veritabanındaki belirtilen tabloya yazar.

    Parameters
    ----------
    df : pd.DataFrame
        Kaydedilecek veri.
    db_path : Path
        SQLite veritabanı dosyasının yolu. Yoksa otomatik oluşturulur.
    table : str
        Hedef tablo adı.
    if_exists : str, optional
        Tablo zaten varsa ne yapılacağı: 'replace' (varsayılan) | 'append' | 'fail'.
    """
    logger.info("Veritabanına bağlanılıyor: %s", db_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(name=table, con=conn, if_exists=if_exists, index=True)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    logger.info(
        "Tablo '%s' başarıyla oluşturuldu → Toplam kayıt sayısı: %d",
        table,
        row_count,
    )


def verify_db(db_path: Path, table: str) -> None:
    """Veritabanı tablosunun doğruluğunu kontrol eder ve özet bilgi basar.

    Parameters
    ----------
    db_path : Path
        SQLite veritabanı dosyasının yolu.
    table : str
        Doğrulanacak tablo adı.
    """
    logger.info("Doğrulama başlıyor: '%s' tablosu kontrol ediliyor…", table)
    with sqlite3.connect(db_path) as conn:
        # Tablo şeması
        schema = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
        # İlk 5 satır önizleme
        preview = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)

    logger.info(
        "Tablo şeması:\n%s",
        schema[["name", "type"]].to_string(index=False),
    )
    logger.info("İlk 5 satır önizleme:\n%s", preview.to_string(index=False))


# ---------------------------------------------------------------------------
# Ana iş akışı
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Ham veri aktarım sürecini baştan sona çalıştırır."""
    logger.info("=" * 60)
    logger.info("Veritabanı kurulum ve ham veri aktarımı başlıyor.")
    logger.info("=" * 60)

    # 1. CSV'yi oku
    df = load_csv(CSV_PATH)

    # 2. SQLite'a kaydet
    save_to_db(df, db_path=DB_PATH, table=TABLE_NAME, if_exists="replace")

    # 3. Doğrula
    verify_db(DB_PATH, table=TABLE_NAME)

    logger.info("=" * 60)
    logger.info("Pipeline tamamlandı. Veritabanı: %s", DB_PATH)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
