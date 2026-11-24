# CLAUDE.md

Konteks proyek untuk Claude Code. Setup environment ada di `README.md`. Semua prosa (markdown, komentar, laporan) ditulis dalam bahasa Indonesia, sedangkan identifier kode dalam bahasa Inggris.

## Tujuan dan keputusan desain
- Tugasnya adalah **lokalisasi root-cause service**: kelas = service di dalam kasus, bukan klasifikasi jenis fault.
- Pipeline (`report/Pipeline.png`): pre-training SSL (TS2Vec, TimeMAE, RUN), lalu fine-tuning dengan mode FineLast (encoder beku + head) dan FineAll.
- Eksperimen: EX-1 one-to-one (pretrain dan evaluasi di dataset yang sama), EX-2..EX-4 one-to-many / transfer, masing-masing ± PCA.
- Urutan data: RE1 → RE2 → RE3. ART (D1) masuk tetapi dikerjakan paling akhir.
- **Skema fitur bersama** (`report/skema_fitur_metrik.md`): K=5 kanal per service (`cpu, mem, latency-50, latency-90, workload`), dan encoder bekerja per service pada `[T, K]`. Input encoder adalah `2*K`, yaitu nilai + mask kanal. Tujuannya supaya encoder bisa ditransfer lintas RE dan sistem.
- 50 kasus RE1-ob dengan skema lama (`latency`/`load`) tetap dipakai, dan kanal yang hilang di-mask. Dua kasus rusak dikeluarkan (`EXCLUDED_CASES` di `data.py`).
- Pakai `src/ssl_rcl/data.py` (`build_samples`, `stratified_split` 60/20/20 seed 42, `make_pretrain_windows`) sebagai loader tunggal. Pre-training hanya memakai split train.

## Evaluasi
- **AC@1 adalah metrik utama.** AC@3 dan AC@5 jenuh oleh prior karena root-cause hanya 2–5 service per sistem (`scripts/diagnose_prior.py`).
- Selalu bandingkan dengan baseline heuristik |z| atas kanal aktif (`baseline.py`). AC@1 baseline per sistem 0.57–0.89, terlemah di tt dan fault delay/loss.
- Laporkan mean ± SD dari 3 seed (0, 1, 2), plus varian scratch (encoder acak) sebagai kontrol.

## Status (per 2026-11-17)
- EX-1 TS2Vec selesai untuk RE1, RE2, dan RE3 (`results/ex1/`, `03_ex1_ringkasan.ipynb`). AC@1: RE1 ±0.92–0.95, RE2 ±0.86–0.89, dan RE3 1.0 (ceiling, n_test 18). Heuristik 0.81 / 0.80 / 0.67.
- Diagnostik: kalau anomali dihapus, AC@1 ≈ acak, jadi tidak ada kebocoran.
- Temuan penting: **pretrain ≈ scratch** in-domain, jadi SSL belum terbukti membantu. Bukti manfaat SSL harus datang dari eksperimen transfer (EX-2+).
- EX-1 TimeMAE selesai (`results/ex1/*_timemae/`, 3 seed). AC@1 pretrain vs scratch (FineLast/FineAll): RE1 0.916/0.929 vs 0.902/0.916, RE2 0.867/0.874 vs 0.852/0.852, RE3 0.963/1.0 vs 0.981/1.0. Sama seperti TS2Vec: pretrain ≈ scratch in-domain. Catatan: suku loss keragaman codeword (`--tm-gamma`) bukan dari paper, ditambahkan karena tokenizer kolaps ke 1 codeword tanpa itu. Patch 12 harus membagi `inject_offset` 300 dan T 540.
- EX-2 TS2Vec ada (`run_ex2.py`, `report/ex2_ts2vec.md`). `run_ex2.py --model timemae` sudah didukung; EX-2 TimeMAE belum dijalankan. Varian ± PCA belum ada.
- Berikutnya: RUN di `src/ssl_rcl/models/` dengan antarmuka yang sama seperti `ts2vec.py` (encoder per service: `encoder(x[B, T, 2K], mask="all_true") -> [B, T, D]`, yang di-pool max pra/pasca injeksi oleh `finetune.service_features`; `run_ex1.py --model` perlu ditambah pilihannya). Paper ada di `reference/`.

## Gotcha
- Di Apple Silicon (MPS), pre-training TS2Vec harus di CPU karena memori bocor akibat panjang random crop yang berubah-ubah. Fine-tuning boleh di MPS.
- Mesin lama adalah M1 8 GB, sehingga FineAll lambat (±400–500 detik per run). Perhatikan memori saat memperbesar batch.
- `cache/` dan `checkpoints/` tidak ada di git. Cache dibuat ulang otomatis, sedangkan checkpoint perlu dilatih ulang atau disalin manual.
- Notebook mengasumsikan cwd = root repo.
