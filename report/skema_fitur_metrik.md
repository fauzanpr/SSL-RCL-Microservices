# Analisis Kolom Metrik RE1–RE3 dan Usulan Skema Fitur

Dihasilkan dari `analysis/metric_schema_audit.py`. Tabel lengkap ada di `report/metric_audit/`.

## 1. Temuan

### Format kolom
Semua kolom metrik mengikuti pola `<service>_<jenis_metrik>` ditambah kolom `time`. Interval sampling 1 detik di semua dataset.

### Jenis metrik (proporsi kasus yang memilikinya)

| Metrik | RE1 ob | RE1 ss | RE1 tt | RE2 (ob/ss/tt) | RE3 (ob/ss/tt) | Catatan |
|---|---|---|---|---|---|---|
| cpu | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | |
| mem | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | |
| latency-50 | 0.6 | 1.0 | 1.0 | 1.0 | 1.0 | NaN ±20% di RE3 |
| latency-90 | 0.6 | 1.0 | 1.0 | 1.0 | 1.0 | NaN ±20% di RE3 |
| workload | 0.6 | 1.0 | 1.0 | 1.0 | 1.0 | |
| error | 1.0 | 1.0 | 1.0 | 0.97 / 0.74 / 0.99 | 1.0 | 40–88% kolom konstan (umumnya 0) |
| diskio | – | – | – | 1.0 | 1.0 | hanya RE2/RE3 |
| socket | – | – | – | 1.0 | 1.0 | hanya RE2/RE3; ±40% konstan |
| latency, load | 0.4 | – | – | – | – | skema lama, hanya di 50 kasus RE1-ob |

Metrik inti yang tersedia di ketiga RE adalah **cpu, mem, latency-50, latency-90, workload**. `error` hanya muncul di sebagian service dan sebagian besar kolomnya konstan.

### RE1-ob memakai dua skema
- 50 kasus (fault cpu dan mem, 5 service) memakai skema lama: `latency` dan `load` (tanpa `latency-50`, `latency-90`, `workload`). Durasinya 4200 detik, dengan 2100 detik sebelum injeksi.
- 75 kasus lain memakai skema baru, dengan durasi 720 detik.
- `latency` (median ±0.05) tidak setara dengan `latency-50` (median ±0.003), jadi kolom ini **tidak boleh** dipetakan begitu saja.
- Skala `cpu` di RE1-ob (median ±4) juga berbeda dari dataset lain (<1). Karena itu normalisasi harus dilakukan per series.

### Service
- Setiap sistem punya service inti (yang memiliki `cpu` secara stabil):
  - ob: 11 service, sama di RE1–RE3 (RE1 punya tambahan `main`).
  - ss: 15 service, identik di RE1–RE3.
  - tt: 64 service sama di ketiga RE. RE2/RE3 menambah 4 database (`ts-food-mysql`, `ts-preserve-mongo`, `ts-preserve-other-mongo`, `ts-notification-mongo`).
- Ada kolom sisa yang bukan service aplikasi, misalnya `istio-init`, `frontend-check`, `PassthroughCluster*`, `InboundPassthroughClusterIpv4*`, dan pod sock-shop yang muncul di RE3-ob (hanya mem/socket). Kolom ini dibuang.
- Root-cause service selalu ada di kolom metrik (735/735 kasus).

### Durasi dan injeksi

| | Panjang (detik) | Pra-injeksi | Pasca-injeksi |
|---|---|---|---|
| RE1 ob | 720 (atau 4200 untuk skema lama) | 360 / 2100 | 360 / 2100 |
| RE1 ss / tt | 720 / 960 | 360 / 480 | 360 / 480 (min 246 di ss) |
| RE2 | 1440 | 720 | 720 |
| RE3 ob/ss / tt | 1440 / 1800 | 720 / 900 | 720 / 900 |

### Kasus bermasalah
- `re1ob_currencyservice_loss_1`: isi `inject_time.txt` = `16933142`, kemungkinan digitnya terpotong. Data berjalan dari 1693313863 sampai 1693314583.
- `re1ob_productcatalogservice_cpu_3`: hanya berisi 62 detik data, dan waktu injeksinya berada di luar rentang data.
- `re1ob_checkoutservice_cpu_4`: data pasca-injeksi hanya 578 detik (masih cukup panjang).

Usulan: dua kasus pertama **dikeluarkan**, sehingga RE1 menjadi 373 kasus.

### Label
| | Jenis fault | Jumlah root-cause service |
|---|---|---|
| RE1 | cpu, mem, disk, delay, loss (seimbang, 75 per kelas) | 5 per sistem |
| RE2 | + socket (seimbang, 45 per kelas) | 5 per sistem |
| RE3 | f1–f5, fault level kode (26/13/26/19/6) | 4 / 3 / 2 |

## 2. Usulan skema fitur

**Satu kasus direpresentasikan sebagai tensor `[N_service, T, K]` dengan K jenis metrik yang tetap.**

1. **Kanal (K = 5):** `cpu, mem, latency-50, latency-90, workload`, ditambah mask biner untuk kanal yang hilang. Contohnya service database yang tidak punya latency, atau 50 kasus RE1-ob dengan skema lama.
   - Varian **K = 7** (tambah `diskio`, `socket`) hanya untuk eksperimen RE2 ↔ RE3.
   - `error` tidak dipakai karena cakupannya tidak lengkap dan sebagian besar konstan.
2. **Service:** memakai daftar service inti per sistem seperti di atas. Kolom infrastruktur dan pod sisa dibuang.
3. **Encoder bersifat per-service atau channel-independent.** Encoder TS2Vec atau TimeMAE melihat input `[T, K]` per service, sehingga `input_dims = K` sama untuk semua sistem dan semua RE. Inilah yang membuat transfer S → T (EX-2) bisa dilakukan. Dimensi service diperlakukan seperti batch.
4. **Normalisasi:** z-score robust per series (per service per metrik) dengan statistik dari segmen pra-injeksi, ditambah epsilon untuk series konstan. Cara ini menyamakan perbedaan skala (misalnya cpu RE1-ob) dan membuat deviasi setelah injeksi terlihat.
5. **Window:**
   - Pre-training (SSL): sliding window di seluruh series, tanpa label.
   - Fine-tuning dan evaluasi: window `[inject − 300 s, inject + 240 s]`. Batas ini dipilih karena aman untuk semua kasus, mengingat RE1-ss punya data pasca-injeksi minimal 246 detik. Opsional, window bisa di-downsample ke 5 detik.
6. **Head (dilatih ulang di target):**
   - Klasifikasi jenis fault: pooling embedding seluruh service, lalu MLP.
   - Lokalisasi root cause: skor per embedding service, lalu softmax di atas N_service. Pendekatan ini tidak bergantung pada sistem.

   Karena head dilatih ulang di dataset target, label yang berbeda antar RE tidak menjadi masalah. Yang ditransfer hanya encoder, dan encoder hanya membutuhkan ruang input yang sama.

## 3. Konsekuensi untuk pipeline

- **PCA (EX-3/EX-4):** PCA pada kolom mentah tidak bisa ditransfer, karena kolomnya berbeda antar sistem dan antar RE. Ada dua opsi:
  - (a) PCA pada embedding encoder. Dimensinya sama, jadi bisa ditransfer.
  - (b) PCA per dataset, di-fit di target.

  Opsi (a) lebih konsisten dengan skema transfer.
- **RUN** bekerja di level kasus pada graf antar variabel (causal discovery). Representasi `[N_service, T, K]` tetap mempertahankan struktur ini, tetapi integrasinya perlu dicek terhadap implementasi aslinya.
- **EX-2 tanpa ART:** setiap sumber hanya punya 2 target, sehingga total eksperimen berubah dari 48 menjadi 36.

## 4. Keputusan dan implementasi

Keputusan:
- **Tugas:** SSL pre-training, lalu fine-tuning untuk **lokalisasi root-cause service**. Kelas yang dicari adalah service.
- **50 kasus RE1-ob skema lama** tetap dipakai. Kanal latency-50, latency-90, dan workload di-mask; kolom `latency` dan `load` tidak dipakai.
- **ART** masuk ke penelitian, tetapi ditambahkan belakangan.

Implementasi ada di `src/ssl_rcl/data.py`, dengan cache dan sanity check di `scripts/prepare_data.py`. Hasil praproses (K = 5, window [−300 s, +240 s), T = 540):

| | Kasus | N service (ob/ss/tt) | Split train/val/test |
|---|---|---|---|
| RE1 | 373 | 11 / 15 / 64 | 223 / 75 / 75 |
| RE2 | 270 | 11 / 15 / 68 | 165 / 60 / 45 |
| RE3 | 90 | 11 / 15 / 68 | 53 / 19 / 18 |

Catatan:
- Split terstratifikasi per (sistem, root service) dengan seed 42.
- Window pre-training SSL hanya diambil dari split train.
- `re2ob_checkoutservice_cpu_2` hanya punya 209 detik data pasca-injeksi, sehingga 30 detik terakhir di-padding dengan nilai terakhir.

### Baseline heuristik (tanpa pembelajaran)
Service diurutkan berdasarkan rata-rata |z| pasca-injeksi, dihitung atas kanal aktif saja. Penjelasan lengkapnya ada di `01_preprocessing_baseline.ipynb`. Hasilnya:

| | RE1 ob/ss/tt | RE2 ob/ss/tt | RE3 ob/ss/tt |
|---|---|---|---|
| AC@1 | 0.72 / 0.82 / 0.62 | 0.89 / 0.82 / 0.66 | 0.87 / 0.80 / 0.57 |
| AC@3 | 0.98 / 1.00 / 0.81 | 0.98 / 0.99 / 0.80 | 1.00 / 0.97 / 0.87 |

Catatan: versi awal membagi dengan K = 5 penuh dan menghasilkan angka lebih tinggi (hingga 0.93). Varian itu diam-diam menguntungkan service yang punya banyak kanal, dan root service di dataset ini selalu termasuk kelompok tersebut. Jadi itu bias struktural, sehingga tidak dipakai.

Baseline ini (beserta baseline RCA standar dari RCAEval) **wajib dilaporkan sebagai pembanding**. Ruang perbaikan terbesar bagi model SSL ada di:
- **Train Ticket**, dengan AC@1 0.57–0.66.
- Fault **delay dan loss**: AC@1 di RE1 hanya 0.12–0.68, dan RE1-tt loss hanya 0.12.

Fault cpu, mem, dan disk hampir selalu benar.

### Confound di RE1-ob
Skema lama di RE1-ob identik dengan fault cpu/mem. Seluruh 49 kasus cpu/mem memakai skema lama (durasi 70 menit), dan seluruh kasus delay/loss/disk memakai skema baru. Untuk lokalisasi service hal ini tidak membocorkan label. Namun, jumlah kanal yang ter-mask berbeda menurut jenis fault, sehingga perlu dicatat di analisis hasil (misalnya dengan melaporkan performa per jenis fault).
