"""
preprocessing.py
----------------
Makale reprodüksiyonu projesi — Adım 2: Veri temizleme ve ön işleme.

Bu modül şu adımları gerçekleştirir:
  1. `heart_disease.db` veritabanından `raw_data` tablosunu okur.
  2. Eksik değerleri (null/NaN) kaldırır.
  3. `age` sütununu gün → yıl cinsine çevirir (÷ 365.25, tam sayıya yuvarlar).
  4. `weight` ve `height` kullanarak `bmi` sütunu türetir; orijinalleri çıkarır.
  5. `ap_hi`, `ap_lo`, `bmi` ve `age` sütunlarına literatür eşik filtresi uygular.
  6. Temizlenmiş veriyi `cleaned_data` tablosu olarak aynı veritabanına yazar.

Literatür eşik değerleri
  - age    : 18 – 80 yıl         (kardiyovasküler çalışma normu)
  - ap_hi  : 60 – 250 mmHg       (JNC 8 / ACC/AHA 2017)
  - ap_lo  : 40 – 150 mmHg       (JNC 8 / ACC/AHA 2017)
  - ap_hi  > ap_lo               (fizyolojik zorunluluk)
  - bmi    : 10 – 60 kg/m²       (WHO aralığı)

Kullanım:
  python preprocessing.py
"""

import sqlite3
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "heart_disease.db"
SRC_TABLE  = "raw_data"
DST_TABLE  = "cleaned_data"

# Literatüre dayalı aykırı değer eşikleri (dahil)
OUTLIER_BOUNDS: dict[str, tuple[float, float]] = {
    "age"  : (18,  80),    # yıl
    "ap_hi": (60,  250),   # mmHg — sistolik
    "ap_lo": (40,  150),   # mmHg — diastolik
    "bmi"  : (10,  60),    # kg/m²
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
# Adım fonksiyonları
# ---------------------------------------------------------------------------

def load_raw_data(db_path: Path, table: str) -> pd.DataFrame:
    """SQLite veritabanından ham veriyi okur.

    Parameters
    ----------
    db_path : Path
        SQLite veritabanı dosyasının yolu.
    table : str
        Okunacak tablo adı.

    Returns
    -------
    pd.DataFrame
        Ham veri içeren DataFrame.

    Raises
    ------
    FileNotFoundError
        Veritabanı dosyası bulunamazsa fırlatılır.
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"Veritabanı bulunamadı: {db_path}")

    logger.info("Veritabanından okunuyor: %s → tablo: '%s'", db_path, table)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn, index_col="id")

    logger.info("Ham veri yüklendi → %d satır, %d sütun", len(df), len(df.columns))
    return df


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik değer (null/NaN) içeren satırları kaldırır.

    Parameters
    ----------
    df : pd.DataFrame
        Girdi DataFrame.

    Returns
    -------
    pd.DataFrame
        Eksik değerlerden arındırılmış DataFrame.
    """
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    logger.info(
        "Eksik değer temizleme: %d satır kaldırıldı → kalan: %d",
        dropped,
        len(df),
    )
    return df


def convert_age_to_years(df: pd.DataFrame) -> pd.DataFrame:
    """'age' sütununu gün cinsinden yıl cinsine çevirir.

    Formül: yıl = round(gün / 365.25)  → tam sayı (Int64)

    Parameters
    ----------
    df : pd.DataFrame
        'age' sütunu gün cinsinden olan DataFrame.

    Returns
    -------
    pd.DataFrame
        'age' sütunu yıl cinsinden olan DataFrame.
    """
    df = df.copy()
    df["age"] = (df["age"] / 365.25).round().astype("Int64")
    logger.info(
        "Yaş dönüşümü tamamlandı: min=%d yıl, max=%d yıl, ortalama=%.1f yıl",
        df["age"].min(),
        df["age"].max(),
        df["age"].mean(),
    )
    return df


def compute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """BMI hesaplar, 'weight' ve 'height' sütunlarını çıkarır.

    Formül: BMI = weight(kg) / (height(m))²

    Parameters
    ----------
    df : pd.DataFrame
        'weight' (kg) ve 'height' (cm) sütunları olan DataFrame.

    Returns
    -------
    pd.DataFrame
        'bmi' sütunu eklenmiş, 'weight' ve 'height' çıkarılmış DataFrame.
    """
    df = df.copy()
    height_m = df["height"] / 100.0          # cm → m
    df["bmi"] = (df["weight"] / height_m**2).round(2)
    df = df.drop(columns=["weight", "height"])
    logger.info(
        "BMI hesaplandı: min=%.2f, max=%.2f, ortalama=%.2f",
        df["bmi"].min(),
        df["bmi"].max(),
        df["bmi"].mean(),
    )
    logger.info("'weight' ve 'height' sütunları çıkarıldı.")
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Literatür eşik değerlerine göre aykırı gözlemleri filtreler.

    Uygulanan filtreler (OUTLIER_BOUNDS sözlüğünden):
      - Her sütun için [alt_eşik, üst_eşik] aralığı dışındaki satırlar silinir.
    Ek fizyolojik kontrol:
      - Sistolik (ap_hi) > Diastolik (ap_lo) olmalıdır.

    Parameters
    ----------
    df : pd.DataFrame
        Girdi DataFrame.

    Returns
    -------
    pd.DataFrame
        Aykırı değerlerden arındırılmış DataFrame.
    """
    df = df.copy()
    before = len(df)

    for col, (lo, hi) in OUTLIER_BOUNDS.items():
        if col not in df.columns:
            logger.warning("Sütun '%s' bulunamadı, atlanıyor.", col)
            continue
        cnt_before = len(df)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        removed = cnt_before - len(df)
        logger.info(
            "  %-6s  [%s, %s]  → %d aykırı satır kaldırıldı",
            col, lo, hi, removed,
        )

    # Fizyolojik zorunluluk: sistolik > diastolik
    cnt_before = len(df)
    df = df[df["ap_hi"] > df["ap_lo"]]
    logger.info(
        "  ap_hi > ap_lo filtresi → %d aykırı satır kaldırıldı",
        cnt_before - len(df),
    )

    total_removed = before - len(df)
    logger.info(
        "Aykırı değer filtresi tamamlandı: toplam %d satır kaldırıldı → kalan: %d",
        total_removed,
        len(df),
    )
    return df


def save_cleaned_data(
    df: pd.DataFrame,
    db_path: Path,
    table: str,
    if_exists: str = "replace",
) -> None:
    """Temizlenmiş veriyi SQLite veritabanına yazar.

    Parameters
    ----------
    df : pd.DataFrame
        Kaydedilecek temizlenmiş veri.
    db_path : Path
        Hedef SQLite veritabanı dosyasının yolu.
    table : str
        Hedef tablo adı.
    if_exists : str, optional
        Tablo varsa: 'replace' (varsayılan) | 'append' | 'fail'.
    """
    logger.info("Temizlenmiş veri '%s' tablosuna yazılıyor…", table)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(name=table, con=conn, if_exists=if_exists, index=True)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    logger.info(
        "Tablo '%s' başarıyla kaydedildi → %d kayıt",
        table,
        row_count,
    )


def verify_cleaned_data(db_path: Path, table: str) -> None:
    """Kaydedilen temizlenmiş tablonun şemasını ve örnek satırlarını loglar.

    Parameters
    ----------
    db_path : Path
        SQLite veritabanı dosyasının yolu.
    table : str
        Doğrulanacak tablo adı.
    """
    with sqlite3.connect(db_path) as conn:
        schema  = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
        preview = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
        stats   = pd.read_sql_query(
            f"SELECT COUNT(*) AS rows, "
            f"MIN(age) AS age_min, MAX(age) AS age_max, "
            f"ROUND(AVG(bmi), 2) AS bmi_mean "
            f"FROM {table}",
            conn,
        )

    logger.info(
        "Tablo şeması:\n%s",
        schema[["name", "type"]].to_string(index=False),
    )
    logger.info("İlk 5 satır:\n%s", preview.to_string(index=False))
    logger.info("Özet istatistikler:\n%s", stats.to_string(index=False))


# ---------------------------------------------------------------------------
# Ana iş akışı
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Veri temizleme sürecini baştan sona çalıştırır."""
    logger.info("=" * 60)
    logger.info("Veri ön işleme (preprocessing) başlıyor.")
    logger.info("=" * 60)

    # 1. Ham veriyi yükle
    df = load_raw_data(DB_PATH, SRC_TABLE)

    # 2. Eksik değerleri temizle
    df = drop_missing(df)

    # 3. Yaşı gün → yıl'a çevir
    df = convert_age_to_years(df)

    # 4. BMI hesapla; weight & height'ı çıkar
    df = compute_bmi(df)

    # 5. Aykırı değerleri filtrele
    logger.info("Aykırı değer filtreleme başlıyor:")
    df = filter_outliers(df)

    # 6. Temizlenmiş veriyi kaydet
    save_cleaned_data(df, DB_PATH, DST_TABLE, if_exists="replace")

    # 7. Doğrula
    verify_cleaned_data(DB_PATH, DST_TABLE)

    logger.info("=" * 60)
    logger.info("Preprocessing tamamlandı. Tablo: '%s' @ %s", DST_TABLE, DB_PATH)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
