# SSL-RCL: Transfer Learning SSL untuk Root Cause Localization Microservice

Riset pre-training self-supervised (TS2Vec, TimeMAE, RUN) pada metrik RCAEval (RE1–RE3, lalu ART), kemudian fine-tuning untuk lokalisasi root-cause **service**. Alur lengkap ada di `report/Pipeline.png`, dan skema fitur di `report/skema_fitur_metrik.md`.

## Setup di device baru

```bash
# 1. Clone dan masuk ke folder proyek
git clone <url-repo> ssl-rcl && cd ssl-rcl

# 2. Environment (Python 3.11 teruji)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
# GPU NVIDIA: pasang torch sesuai versi CUDA dulu, contoh:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m ipykernel install --user --name ssl-rcl

# 3. Dataset RCAEval (default hanya metrics.parquet + inject_time.txt, ±270 MB)
hf auth login                      # kalau diminta (atau: export HF_TOKEN=...)
python scripts/download_dataset.py
#   atau simpan di luar repo, symlink ./dataset dibuat otomatis:
#   python scripts/download_dataset.py --data-dir ~/datasets/rcaeval

# 4. Bangun cache praproses (cache/RE*_K5_pre300_post240.pt) dan cek kewajaran data
python scripts/prepare_data.py --re RE1 RE2 RE3
```

Output yang diharapkan di langkah 3 adalah `RE1: 373`, `RE2: 270`, dan `RE3: 90` kasus valid. Dua kasus RE1 yang rusak memang dikeluarkan.

### Yang tidak ikut di git
| Folder | Ukuran | Cara mendapatkan di device baru |
|---|---|---|
| `dataset/` | ±270 MB (metrik) | `scripts/download_dataset.py` |
| `cache/` | ±300 MB | otomatis dibuat ulang (langkah 4, atau saat script pertama jalan) |
| `checkpoints/ex1/` | ±21 MB (zip) | salin manual, lihat bagian di bawah |

Hasil (`results/`), log, notebook, dan laporan ikut di git.

### Memindahkan checkpoint TS2Vec (EX-1)
Isinya 9 file `checkpoints/ex1/<RE>_ts2vec_seed<s>.pt` (RE1–RE3 × seed 0–2), yaitu encoder TS2Vec hasil pre-training. Checkpoint ini dipakai oleh `scripts/diagnose_prior.py` dan oleh eksperimen transfer (EX-2+), yang memakai encoder RE1 untuk fine-tuning di RE2/RE3. Melatih ulang butuh ±20 menit per seed di CPU, atau ±3 jam untuk semuanya lewat `run_ex1.py`, dan itu juga menimpa `results/ex1/`.

Salin saja secara manual. WhatsApp, Google Drive, flash disk, atau AirDrop semuanya bisa:

```bash
# device lama (dari root repo)
zip -r checkpoints_ex1.zip checkpoints/ex1

# device baru: taruh zip di root repo, lalu
unzip checkpoints_ex1.zip          # hasilnya checkpoints/ex1/*.pt
```

Di WhatsApp, kirim sebagai **Dokumen**, bukan foto/media, supaya file tidak diubah. Zip-nya sudah di-ignore git.

`cache/` (±300 MB) juga boleh disalin dengan cara yang sama supaya tidak perlu menjalankan `prepare_data.py`. Tapi cache dibangun ulang otomatis dari dataset, jadi tidak wajib disalin.

### Catatan device
- `run_ex1.py` memilih device secara otomatis: `mps` → `cuda` → `cpu`. Pre-training memakai `--pretrain-device cpu` secara default, karena di MPS terjadi kebocoran memori akibat panjang random crop TS2Vec yang berubah-ubah. Di CUDA, coba `--pretrain-device cuda`.
- Notebook mengasumsikan working directory = root repo (`ROOT = Path.cwd()`), jadi jalankan `jupyter lab` dari root.

## Menjalankan ulang EX-1

```bash
python scripts/run_ex1.py --source RE1   # juga RE2, RE3; hasil ke results/ex1/<S>_ts2vec/
python scripts/run_ex1.py --source RE1 --model timemae   # hasil ke results/ex1/<S>_timemae/ (CUDA dipilih otomatis)
python scripts/run_ex1.py --source RE1 --model run       # hasil ke results/ex1/<S>_run/ (pre-training RUN ±20 detik di GPU)
python scripts/diagnose_prior.py         # uji prior / anomali dihapus
```

### Menjalankan di background (RE1 → RE2 → RE3, 3 seed)
```bash
python scripts/launch_ex1.py                    # timemae; tetap jalan walau terminal ditutup
python scripts/check_ex1.py --watch 30          # pantau progress, GPU, dan hasil sementara
python scripts/launch_ex1.py --stop             # hentikan
```
Log di `logs/<exp>_<RE>_<model>.log`, status di `logs/<exp>_<model>_status.json`. Argumen setelah `--` diteruskan ke `run_ex1.py`, misalnya `-- --n-iters 5000`.

Untuk EX-2 tambahkan `--exp ex2` (`--sources` di sini berarti daftar TARGET, default RE1 RE2):
```bash
python scripts/launch_ex1.py --exp ex2 -- --fractions 0.1 0.25 0.5 1.0 --modes FineLast
python scripts/check_ex1.py --exp ex2 --watch 30
```

## EX-2 (transfer lintas RE)

Memakai checkpoint EX-1 sebagai encoder sumber, jadi tidak ada pre-training ulang. Script berjalan per target dan membandingkan varian `transfer` (encoder dari RE lain), `indomain` (encoder dari target sendiri), dan `scratch` (encoder acak) pada split dan subset label yang sama.

```bash
python scripts/run_ex2.py --target RE2                                   # sumber: RE1 dan RE3
python scripts/run_ex2.py --target RE2 --modes FineLast --fractions 0.1 0.25 0.5 1.0
python scripts/run_ex2.py --target RE2 --modes FineAll --resume          # lanjutkan run yang terhenti
```

`--fractions` men-subsample split train target (terstratifikasi, minimal 1 kasus per root service) untuk skenario label sedikit, sedangkan val dan test tetap penuh. Hasil disimpan di `results/ex2/<target>_<model>/`. Tambahkan `--model timemae` untuk TimeMAE (arsitektur encoder dibaca dari `results/ex1/<RE>_timemae/config.json`). RE3 sebagai target sudah mencapai ceiling (AC@1 ≈ 1.0 bahkan untuk scratch), jadi kurang informatif.

## Struktur
```
src/ssl_rcl/        data.py (loader + skema K=5), baseline.py (heuristik |z|), finetune.py (FineLast/FineAll)
src/ssl_rcl/models/ ts2vec.py, timemae.py, run.py
scripts/            download_dataset.py, prepare_data.py, run_ex1.py, run_ex2.py, diagnose_prior.py,
                    paired_test.py, launch_ex1.py + check_ex1.py (background, ex1 dan ex2)
analysis/           metric_schema_audit.py -> report/metric_audit/
results/ex1/        summary.csv, predictions.csv, history.csv per RE
results/ex2/        idem, per target (kolom tambahan: variant, source, fraction)
reference/          paper TS2Vec, TimeMAE, RUN
0x_*.ipynb          preprocessing+baseline, EX-1 per RE, ringkasan
```
