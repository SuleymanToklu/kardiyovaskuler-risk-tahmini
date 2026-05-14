"""
eda.py
------
Makale reprodüksiyonu projesi — Adım 3: Keşifsel Veri Analizi (EDA).

Bu modül `cleaned_data` tablosundan istatistiksel özet ve korelasyon matrisini
hesaplayarak aynı veritabanına yazar. app.py bu tablolardan grafik üretir.

Oluşturulan tablolar:
  - eda_stats       : Her sütun için temel istatistikler (describe çıktısı).
  - eda_correlation : Pearson korelasyon matrisi (geniş format).

Kullanım:
  python eda.py
"""

import sqlite3
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "heart_disease.db"
SRC_TABLE = "cleaned_data"

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
# Adım fonksiyonları
# ---------------------------------------------------------------------------

def load_data(db_path: Path, table: str) -> pd.DataFrame:
    """SQLite'tan temizlenmiş veriyi okur."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Veritabanı bulunamadı: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn, index_col="id")
    logger.info("Yüklendi: %d satır, %d sütun. Sütunlar: %s",
                len(df), len(df.columns), df.columns.tolist())
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Her sütun için temel istatistikleri hesaplar ve eksik değer sayısını ekler.

    Returns
    -------
    pd.DataFrame
        column, count, mean, std, min, pct_25, pct_50, pct_75, max, missing sütunları.
    """
    stats = df.describe().T
    stats.index.name = "column"
    stats = stats.reset_index()
    stats.columns = [
        c.replace("%", "pct").replace(" ", "_") for c in stats.columns
    ]
    stats["missing"] = df.isnull().sum().values
    stats["n_unique"] = [df[col].nunique() for col in df.columns]
    logger.info("Özet istatistikler hesaplandı: %d sütun", len(stats))
    return stats


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson korelasyon matrisini geniş formatta hesaplar.

    Returns
    -------
    pd.DataFrame
        İlk sütun 'column', geri kalanlar diğer sütunlarla korelasyonlar.
    """
    corr = df.corr(numeric_only=True).round(4)
    corr.index.name = "column"
    result = corr.reset_index()
    logger.info(
        "Korelasyon matrisi hesaplandı: %d × %d",
        len(result), len(result.columns) - 1,
    )
    return result


def save_to_db(df: pd.DataFrame, db_path: Path, table: str) -> None:
    """DataFrame'i SQLite'a yazar."""
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
    logger.info("'%s' tablosuna kaydedildi → %d satır", table, len(df))


# ---------------------------------------------------------------------------
# Ana iş akışı
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """EDA adımlarını baştan sona çalıştırır."""
    logger.info("=" * 60)
    logger.info("EDA (Keşifsel Veri Analizi) başlıyor.")
    logger.info("=" * 60)

    df = load_data(DB_PATH, SRC_TABLE)

    logger.info("─" * 40)
    logger.info("Özet istatistikler hesaplanıyor…")
    stats = compute_summary_stats(df)
    save_to_db(stats, DB_PATH, "eda_stats")

    logger.info("─" * 40)
    logger.info("Korelasyon matrisi hesaplanıyor…")
    corr = compute_correlation(df)
    save_to_db(corr, DB_PATH, "eda_correlation")

    logger.info("=" * 60)
    logger.info("EDA tamamlandı.")
    logger.info("Tablolar: eda_stats, eda_correlation @ %s", DB_PATH)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
