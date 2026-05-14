"""
run_all.py
----------
Makale reprodüksiyonu projesi — Tüm pipeline adımlarını sırayla çalıştırır.

Kullanım:
  python run_all.py
"""

import sys
import logging
import importlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STEPS = [
    ("database_setup", "Adım 1 — Veritabanı Kurulumu"),
    ("preprocessing",  "Adım 2 — Veri Ön İşleme"),
    ("eda",            "Adım 3 — EDA (Keşifsel Veri Analizi)"),
    ("fuzzification",  "Adım 4 — Bulanıklaştırma"),
    ("model_training", "Adım 5 — Model Eğitimi"),
]


def run_step(module_name: str, label: str) -> None:
    logger.info("─" * 60)
    logger.info("▶  %s", label)
    logger.info("─" * 60)
    mod = importlib.import_module(module_name)
    mod.run_pipeline()


def main() -> None:
    logger.info("=" * 60)
    logger.info("  TAM PİPELINE BAŞLIYOR")
    logger.info("=" * 60)

    for module_name, label in STEPS:
        try:
            run_step(module_name, label)
        except Exception as exc:
            logger.error("HATA (%s): %s", label, exc, exc_info=True)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("  TÜM ADIMLAR TAMAMLANDI")
    logger.info("=" * 60)
    logger.info("  Arayüzü başlatmak için: streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
