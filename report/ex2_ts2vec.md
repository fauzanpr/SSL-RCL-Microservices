# TS2Vec untuk lokalisasi root-cause: EX-1 dan EX-2

Tanggal: 2026-11-18. Dokumen ini adalah bacaan pendamping untuk panduan slide `report/ts2vec_ex1_ex2_ppt.md`. Urutan bagiannya sama dengan urutan slide, dan setiap judul bagian mencantumkan nomor slide yang sesuai. Slide berisi intinya, sedangkan dokumen ini berisi angka lengkap dan penjelasannya.

## Ringkasan singkat

- Model berbasis TS2Vec menemukan root-cause service jauh lebih baik daripada heuristik sederhana. Uji pengecekan juga menunjukkan model benar-benar membaca anomali di metrik, bukan menghafal service.
- **EX-1** (pre-training dan pengujian di dataset yang sama): encoder hasil pre-training **tidak lebih baik** daripada encoder acak. Keunggulan model atas heuristik berasal dari fine-tuning, bukan dari pre-training.
- **EX-2** (encoder dipindah ke dataset lain): encoder yang di-pre-train di **RE1** lalu dipakai untuk **RE2** sedikit lebih baik daripada encoder acak. AC@1 naik sekitar 0.04–0.07, atau 2–3 dari 45 kasus uji. Peningkatan ini lolos uji statistik di 4 dari 5 pengaturan dengan data latih 25% ke atas.
- Efek EX-2 **tidak berlaku umum**:
  - Encoder dari RE3 tidak membantu RE2.
  - Di RE1 sebagai dataset tujuan, tidak ada encoder pre-training yang lebih baik daripada encoder acak.
  - Saat label sangat sedikit (10% data latih), semua encoder setara.

---

## Bagian A: Latar

### A.1 Masalah dan tujuan (Slide 1)

Saat terjadi gangguan di sistem microservice, operator perlu tahu **service mana yang menjadi sumber masalah** (root-cause). Riset ini memperlakukannya sebagai tugas memilih satu service dari semua service di dalam satu kasus gangguan. Tugasnya bukan menebak jenis gangguannya.

Data berasal dari benchmark **RCAEval**: RE1, RE2, dan RE3. Setiap dataset berisi tiga sistem microservice, yaitu Online Boutique (`ob`), Sock Shop (`ss`), dan Train Ticket (`tt`). Setiap kasus berisi metrik per service di sekitar waktu gangguan disuntikkan, dan label service mana yang diganggu.

Pertanyaan riset: **apakah pre-training tanpa label (self-supervised learning) membantu menemukan root-cause?** Pre-training seperti ini menarik karena metrik tanpa label mudah dikumpulkan, sedangkan kasus berlabel mahal.

### A.2 Cara kerja dan pembanding (Slide 2)

#### Data dan pembagiannya

Setiap dataset dibagi menjadi data latih, data validasi, dan data uji (60/20/20). Pembagian dilakukan merata per sistem dan per root service, dengan seed 42. Pembagian yang sama dipakai di EX-1 dan EX-2.

| Dataset | Total kasus | Data latih | Data validasi | Data uji | Nilai 1 kasus uji di AC@1 |
|---|---|---|---|---|---|
| RE1 | 373 | 223 | 75 | 75 | 0.013 |
| RE2 | 270 | 165 | 60 | 45 | 0.022 |
| RE3 | 90 | 53 | 19 | 18 | 0.056 |

Dua kasus RE1 yang datanya rusak dikeluarkan (`EXCLUDED_CASES` di `src/ssl_rcl/data.py`).

#### Input model

Setiap service diwakili oleh **5 metrik**: CPU, memori, latensi p50, latensi p90, dan beban (workload). Jika suatu metrik tidak tersedia di sebuah dataset, metrik itu ditandai sebagai kosong (di-mask), sehingga semua dataset memakai format yang sama. Format yang sama inilah yang memungkinkan encoder dipindah antar dataset di EX-2.

Untuk setiap kasus, yang diambil adalah 300 detik sebelum gangguan dan 240 detik sesudahnya. Metrik dinormalisasi dengan statistik dari bagian sebelum gangguan.

#### Tiga tahap

1. **Pre-training (tanpa label).** Encoder TS2Vec belajar mengenali pola metrik dari potongan deret waktu (540 detik, digeser 270 detik) selama 600 iterasi. Hanya data latih yang dipakai, tanpa label root-cause.
2. **Fine-tuning (dengan label).** Encoder membaca metrik setiap service. Lapisan terakhir (head) kemudian memberi skor ke setiap service, dan service dengan skor tertinggi dipilih sebagai root-cause. Ada dua cara melatih:
   - **FineLast**: encoder dibekukan dan hanya head yang dilatih. Cara ini paling langsung mengukur kualitas encoder.
   - **FineAll**: encoder dan head dilatih bersama.
3. **Uji** pada data uji dengan metrik **AC@1**, yaitu persentase kasus ketika service dengan skor tertinggi memang root-cause-nya.

AC@3 dan AC@5 juga dihitung, tetapi hampir selalu tinggi karena setiap sistem hanya punya 2–5 kandidat root-cause. Keduanya kurang bisa membedakan model, sehingga tidak dipakai untuk menarik kesimpulan.

#### Pembanding

| Pembanding | Artinya | Kenapa perlu |
|---|---|---|
| **Encoder acak** (`scratch`) | Arsitektur sama, tetapi bobot encoder acak, tanpa pre-training | Memisahkan efek pre-training dari efek fine-tuning. **Pre-training dianggap berguna hanya jika mengalahkan pembanding ini.** |
| **Heuristik** | Pilih service dengan lonjakan metrik (\|z\|) terbesar setelah gangguan, tanpa pembelajaran | Acuan minimal: model harus lebih baik daripada aturan sederhana |

Setiap pengaturan diulang dengan **3 seed** (0, 1, 2) dan dilaporkan sebagai rata-rata ± simpangan baku (SD).

---

## Bagian B: EX-1, dilatih dan diuji di dataset yang sama

Di EX-1, pre-training, fine-tuning, dan pengujian dilakukan di dataset yang sama, misalnya RE1 → RE1. Semua data latih dipakai. Hasil mentah ada di `results/ex1/<dataset>_ts2vec/`, dan ringkasannya di `03_ex1_ringkasan.ipynb`.

### B.1 Hasil EX-1 (Slide 3)

AC@1 pada data uji, rata-rata ± SD dari 3 seed:

| Dataset | Heuristik | Cara melatih | Encoder acak | Pre-training TS2Vec |
|---|---|---|---|---|
| RE1 | 0.813 | FineLast | 0.938 ± 0.008 | 0.924 ± 0.020 |
| RE1 | | FineAll | 0.933 ± 0.023 | 0.947 ± 0.013 |
| RE2 | 0.800 | FineLast | 0.859 ± 0.013 | 0.889 ± 0.022 |
| RE2 | | FineAll | 0.867 ± 0.022 | 0.859 ± 0.013 |
| RE3 | 0.667 | FineLast | 1.000 ± 0.000 | 1.000 ± 0.000 |
| RE3 | | FineAll | 1.000 ± 0.000 | 1.000 ± 0.000 |

Cara membaca:
- **Model jauh lebih baik daripada heuristik.** Selisihnya +0.11 sampai +0.13 di RE1, +0.06 sampai +0.09 di RE2, dan +0.33 di RE3.
- **Pre-training dan encoder acak setara.** Di semua dataset, selisih keduanya hanya 1–2 kasus uji dan arahnya tidak konsisten. Kadang pre-training sedikit lebih tinggi (RE1 FineAll, RE2 FineLast), kadang encoder acak yang lebih tinggi (RE1 FineLast, RE2 FineAll).
- **RE3 sudah mencapai batas atas.** Semua model mendapat 1.000, jadi RE3 tidak bisa membedakan model. Data ujinya juga hanya 18 kasus.

### B.2 Apakah model benar-benar membaca anomali? (Slide 4)

Skor tinggi bisa saja muncul dengan cara yang tidak diinginkan, misalnya model sekadar menghafal service yang paling sering menjadi root-cause. Untuk mengeceknya, dijalankan dua uji (`scripts/diagnose_prior.py`, encoder pre-training seed 0, FineLast):

1. **Tebakan dari frekuensi**: urutkan service menurut seberapa sering ia menjadi root-cause di data latih, tanpa melihat metrik sama sekali.
2. **Anomali dihapus**: bagian metrik setelah gangguan diganti dengan potongan data normal (sebelum gangguan) dari kasus yang sama. Jika model memang membaca anomali, skornya seharusnya jatuh ke tingkat tebakan acak.

| Dataset | Model, data normal | Model, anomali dihapus | Tebakan dari frekuensi | Tebakan acak (1/jumlah service) |
|---|---|---|---|---|
| RE1 | 0.907 | **0.027** | 0.200 | 0.058 |
| RE2 | 0.911 | **0.089** | 0.200 | 0.057 |
| RE3 | 1.000 | **0.000** | 0.500 | 0.056 |

Ada dua kesimpulan dari tabel ini:
- Tanpa anomali, AC@1 jatuh ke tingkat tebakan acak. **Model benar-benar memakai sinyal anomali di metrik.**
- Tebakan dari frekuensi jauh di bawah model (0.20–0.50). Jadi skor tinggi tidak berasal dari menghafal service yang sering menjadi root-cause.

### B.3 Kesimpulan EX-1

Model bekerja dengan benar, karena ia membaca anomali dan mengalahkan heuristik. Namun **pre-training TS2Vec belum terbukti membantu**, karena encoder acak sama baiknya. Salah satu kemungkinan penjelasannya: jika data pre-training dan data fine-tuning berasal dari dataset yang sama, fine-tuning sudah cukup untuk mempelajari polanya sendiri. Karena itu, bukti manfaat pre-training dicari di EX-2, ketika encoder dipindah ke dataset lain.

---

## Bagian C: EX-2, encoder dipindah ke dataset lain

### C.1 Pertanyaan dan cara EX-2 (Slide 5)

Pertanyaan EX-2: **apakah encoder yang di-pre-train di satu dataset (misalnya RE1) berguna untuk dataset lain (misalnya RE2)?** EX-2 juga menguji apakah manfaatnya lebih besar saat kasus berlabel untuk fine-tuning hanya sedikit.

#### Encoder yang dibandingkan

Semua encoder diambil dari checkpoint EX-1, jadi tidak ada pre-training baru. Pre-training hanya memakai data latih dari dataset asalnya, sehingga tidak ada data uji yang bocor.

| Nama di hasil | Artinya |
|---|---|
| `transfer` | Encoder di-pre-train di dataset lain, misalnya dari RE1 untuk diuji di RE2 |
| `indomain` | Encoder di-pre-train di dataset tujuan itu sendiri (sama seperti EX-1) |
| `scratch` | Encoder acak, tanpa pre-training (**pembanding utama**) |
| heuristik | Pilih service dengan lonjakan metrik terbesar |

RE1 dan RE2 dipakai sebagai dataset tujuan. **RE3 tidak dipakai sebagai dataset tujuan** karena semua model, termasuk encoder acak, sudah mencapai AC@1 = 1.000 di EX-1. RE3 tetap dipakai sebagai **sumber** encoder.

#### Jumlah data latih berlabel

Untuk meniru situasi ketika kasus berlabel sedikit, fine-tuning hanya memakai sebagian data latih: 10%, 25%, 50%, atau 100%. Cara ini sama dengan eksperimen *"different proportions of training set"* di paper TimeMAE.

| Dataset tujuan | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| RE1 | 28 kasus | 60 | 118 | 223 |
| RE2 | 15 kasus | 45 | 90 | 165 |

Ada tiga hal yang membuat perbandingan ini adil:
- Kasus diambil merata per sistem dan per root service. Setiap root service selalu punya minimal satu contoh.
- Untuk seed yang sama, semua encoder dilatih dengan kasus yang persis sama.
- Data validasi dan data uji selalu dipakai penuh.

#### Apa saja yang sudah dijalankan

| Dataset tujuan | FineLast | FineAll |
|---|---|---|
| RE2 | 10%, 25%, 50%, 100% | 25%, 50% |
| RE1 | 10%, 25%, 50%, 100% | belum |

Setiap sel berisi 4 encoder × 3 seed = 12 run. FineAll jauh lebih lambat di CPU, sekitar 6–11 menit per run, sedangkan FineLast sekitar 30–40 detik. Karena itu, FineAll baru dijalankan di bagian yang paling menarik, yaitu RE2 dengan data latih 25% dan 50%.

### C.2 Hasil utama EX-2 (Slide 6)

#### Dataset tujuan RE2

AC@1 pada data uji, rata-rata ± SD dari 3 seed. Heuristik mendapat 0.800.

| Data latih | Cara melatih | `indomain` (RE2) | `scratch` | `transfer` dari RE1 | `transfer` dari RE3 |
|---|---|---|---|---|---|
| 10% | FineLast | 0.822 ± 0.022 | 0.830 ± 0.026 | 0.830 ± 0.026 | 0.844 ± 0.022 |
| 25% | FineLast | 0.867 ± 0.000 | 0.844 ± 0.038 | **0.904 ± 0.013** | 0.859 ± 0.034 |
| 25% | FineAll | 0.859 ± 0.013 | 0.852 ± 0.013 | **0.881 ± 0.013** | 0.867 ± 0.000 |
| 50% | FineLast | 0.904 ± 0.013 | 0.844 ± 0.000 | **0.911 ± 0.000** | 0.881 ± 0.013 |
| 50% | FineAll | 0.867 ± 0.022 | 0.844 ± 0.022 | **0.889 ± 0.022** | 0.874 ± 0.013 |
| 100% | FineLast | 0.889 ± 0.022 | 0.859 ± 0.013 | **0.911 ± 0.022** | 0.867 ± 0.000 |

Dengan data latih 25% ke atas, `transfer` dari RE1 selalu menjadi encoder terbaik. Selisihnya terhadap `scratch` sekitar 2–3 kasus uji.

#### Dataset tujuan RE1

AC@1 pada data uji, FineLast, rata-rata ± SD dari 3 seed. Heuristik mendapat 0.813.

| Data latih | `indomain` (RE1) | `scratch` | `transfer` dari RE2 | `transfer` dari RE3 |
|---|---|---|---|---|
| 10% | 0.893 ± 0.013 | 0.867 ± 0.027 | 0.862 ± 0.008 | 0.889 ± 0.020 |
| 25% | 0.902 ± 0.020 | 0.907 ± 0.013 | 0.889 ± 0.015 | 0.902 ± 0.008 |
| 50% | 0.911 ± 0.020 | 0.911 ± 0.008 | 0.902 ± 0.028 | 0.898 ± 0.008 |
| 100% | 0.924 ± 0.020 | 0.924 ± 0.008 | 0.920 ± 0.013 | 0.911 ± 0.008 |

Di RE1, semua encoder berada dalam 1–2 kasus uji satu sama lain. Polanya sama seperti EX-1.

#### Di sistem mana peningkatan RE2 terjadi?

AC@1 dihitung per sistem. Setiap sel menggabungkan 15 kasus uji × 3 seed.

| Data latih | Cara melatih | Sistem | `indomain` | `scratch` | `transfer` dari RE1 | `transfer` dari RE3 |
|---|---|---|---|---|---|---|
| 25% | FineLast | ob | 0.822 | 0.800 | 0.844 | 0.844 |
| 25% | FineLast | tt | 0.778 | 0.733 | **0.867** | 0.733 |
| 50% | FineLast | ob | 0.867 | 0.844 | 0.867 | 0.867 |
| 50% | FineLast | tt | 0.844 | 0.689 | **0.867** | 0.778 |
| 100% | FineLast | ob | 0.844 | 0.800 | 0.889 | 0.822 |
| 100% | FineLast | tt | 0.822 | 0.778 | 0.844 | 0.778 |
| 25% | FineAll | tt | 0.733 | 0.711 | 0.778 | 0.756 |
| 50% | FineAll | tt | 0.756 | 0.711 | 0.800 | 0.756 |

Peningkatan terutama terjadi di **`tt`**, yaitu sistem tersulit bagi semua metode. Sistem `ss` tidak ditampilkan karena semua encoder sudah mencapai 1.000 (satu sel 0.978), sehingga tidak membedakan apa pun.

### C.3 Apakah bukan kebetulan? (Slide 7)

Data uji RE2 hanya berisi 45 kasus, jadi selisih 2–3 kasus bisa saja terjadi karena kebetulan. Untuk mengujinya, dipakai **bootstrap berpasangan** (`scripts/paired_test.py`):

1. Setiap encoder dibandingkan dengan `scratch` pada kasus uji yang sama.
2. Untuk setiap kasus, dicatat apakah tiap model menjawab benar. Hasilnya dirata-rata atas 3 seed, supaya kasus yang sama tidak terhitung tiga kali.
3. Kasus uji diambil ulang secara acak sebanyak 10.000 kali. Dari situ diperoleh rentang 95% untuk selisih AC@1.
4. **Jika rentang itu tidak mencakup 0, selisihnya dianggap bermakna** (bukan kebetulan).

Kolom terakhir menunjukkan berapa pasangan (kasus, seed) yang hanya dijawab benar oleh encoder tersebut, dibandingkan yang hanya dijawab benar oleh `scratch`.

| Data latih | Cara melatih | Encoder | Selisih vs `scratch` | Rentang 95% | Bermakna? | Hanya encoder ini benar / hanya `scratch` benar |
|---|---|---|---|---|---|---|
| 10% | FineLast | `indomain` | −0.007 | [−0.074, +0.059] | tidak | 5 / 6 |
| 10% | FineLast | `transfer` RE1 | 0.000 | [−0.067, +0.059] | tidak | 5 / 5 |
| 10% | FineLast | `transfer` RE3 | +0.015 | [−0.037, +0.074] | tidak | 5 / 3 |
| 25% | FineLast | `indomain` | +0.022 | [−0.030, +0.081] | tidak | 6 / 3 |
| 25% | FineLast | `transfer` RE1 | **+0.059** | [+0.015, +0.126] | **ya** | 8 / 0 |
| 25% | FineLast | `transfer` RE3 | +0.015 | [−0.022, +0.059] | tidak | 5 / 3 |
| 25% | FineAll | `indomain` | +0.007 | [−0.030, +0.052] | tidak | 4 / 3 |
| 25% | FineAll | `transfer` RE1 | +0.030 | [−0.007, +0.089] | tidak | 6 / 2 |
| 25% | FineAll | `transfer` RE3 | +0.015 | [−0.030, +0.074] | tidak | 5 / 3 |
| 50% | FineLast | `indomain` | **+0.059** | [+0.007, +0.126] | **ya** | 8 / 0 |
| 50% | FineLast | `transfer` RE1 | **+0.067** | [+0.015, +0.133] | **ya** | 9 / 0 |
| 50% | FineLast | `transfer` RE3 | +0.037 | [0.000, +0.089] | tidak (batas) | 5 / 0 |
| 50% | FineAll | `indomain` | +0.022 | [−0.007, +0.052] | tidak | 5 / 2 |
| 50% | FineAll | `transfer` RE1 | **+0.044** | [+0.007, +0.096] | **ya** | 7 / 1 |
| 50% | FineAll | `transfer` RE3 | +0.030 | [−0.022, +0.089] | tidak | 6 / 2 |
| 100% | FineLast | `indomain` | +0.030 | [−0.007, +0.081] | tidak | 5 / 1 |
| 100% | FineLast | `transfer` RE1 | **+0.052** | [+0.007, +0.111] | **ya** | 8 / 1 |
| 100% | FineLast | `transfer` RE3 | +0.007 | [−0.030, +0.052] | tidak | 3 / 2 |

Untuk RE1 sebagai dataset tujuan, **tidak ada satu pun perbandingan yang bermakna**. Selisihnya hanya antara −0.018 dan +0.027, dan semua rentang 95% mencakup 0. Hasil lengkap kedua dataset ada di `results/ex2/<dataset>_ts2vec/paired_test.csv`.

Cara membaca hasil ini:
- **Transfer RE1 → RE2 bermakna di 4 dari 5 pengaturan** dengan data latih 25% ke atas. Pengecualiannya FineAll 25%, yang rentangnya sedikit menyentuh 0.
- Di FineLast, hampir tidak ada kasus yang dijawab benar oleh `scratch` tetapi salah oleh `transfer` RE1 (0–1 pasangan). Artinya, encoder ini menambah kasus yang benar tanpa mengorbankan kasus lain.
- **Efeknya kecil.** Batas bawah rentang 95% dekat dengan 0 (+0.007 sampai +0.015).
- **Ada 30 perbandingan**, sehingga 1–2 hasil bisa tampak bermakna hanya karena kebetulan. Karena itu, kesimpulan bertumpu pada pola yang berulang (`transfer` RE1 → RE2 di beberapa pengaturan), bukan pada satu hasil tunggal. Contohnya, `indomain` di FineLast 50% bermakna sendirian tanpa didukung pengaturan lain, jadi sebaiknya tidak diklaim.

### C.4 Yang tidak sesuai harapan (Slide 8)

1. **Pre-training tidak membantu saat label sangat sedikit.** Dengan data latih 10% (15 kasus di RE2, 28 kasus di RE1), semua encoder setara. Ini berlawanan dengan harapan umum dalam penelitian self-supervised, yaitu bahwa pre-training paling berguna saat label sedikit. Kemungkinan dengan kasus sesedikit itu, hasil lebih ditentukan oleh variasi pelatihan head daripada oleh kualitas encoder.

2. **Efek transfer hanya muncul di satu pasangan dataset.**
   - Encoder dari RE3 tidak membantu RE2 secara bermakna. RE3 adalah dataset terkecil (53 kasus latih), jadi encoder-nya kemungkinan belajar paling sedikit.
   - Di RE1 sebagai dataset tujuan, tidak ada encoder pre-training yang lebih baik daripada encoder acak.

   Satu kemungkinan penjelasan: RE1 punya data latih paling banyak (223 kasus), sehingga encoder-nya paling umum. Penjelasan ini belum diuji.

3. **Melatih ulang encoder (FineAll) tidak lebih baik daripada membekukannya (FineLast).** Untuk encoder hasil pre-training, FineAll malah sedikit lebih buruk. Dengan data latih 50%, `indomain` turun dari 0.904 ke 0.867, dan `transfer` RE1 dari 0.911 ke 0.889. Semua run FineAll berhenti di epoch 9–11, karena loss validasi terbaik sudah dicapai di epoch 1–3. Ini tanda model cepat overfit, yaitu terlalu menyesuaikan diri dengan data latih yang sedikit. Akibatnya, ikut melatih encoder justru sedikit merusak hasil pre-training.

---

## Bagian D: Penutup

### D.1 Kesimpulan (Slide 9)

1. **Model berbasis TS2Vec mengalahkan heuristik** (+0.05 sampai +0.33 AC@1, tergantung dataset) dan terbukti membaca anomali di metrik (EX-1, uji anomali dihapus).
2. **Keunggulan itu berasal dari fine-tuning, bukan dari pre-training.** Di EX-1, encoder acak sama baiknya dengan encoder hasil pre-training.
3. **Pre-training baru terlihat membantu saat dipindah dari RE1 ke RE2** (EX-2). Efeknya konsisten tetapi kecil (2–3 kasus uji), dan tidak muncul di pasangan dataset lain maupun saat label sangat sedikit.

Jadi self-supervised learning belum terbukti sebagai faktor kunci. Ada satu sinyal positif (RE1 → RE2) yang perlu diperkuat dengan eksperimen berikutnya.

### D.2 Langkah berikutnya (Slide 10)

1. **Mengulang EX-1 dan EX-2 dengan TimeMAE dan RUN.** `scripts/run_ex2.py` dan `scripts/paired_test.py` sudah menerima argumen `--model`.
2. **Menambahkan varian ± PCA dan dataset ART.**

### D.3 Batasan

- **Data uji kecil.** RE2 punya 45 kasus uji dan RE3 hanya 18, jadi rentang 95% lebar dan efek kecil sulit dipastikan.
- **Banyak perbandingan.** Ada 30 perbandingan di uji statistik EX-2. Lihat penjelasannya di C.3.
- **Penyebab keberhasilan RE1 → RE2 belum diketahui.** Dugaan bahwa ukuran data pre-training RE1 yang lebih besar adalah penyebabnya (lihat C.4) belum diuji, dan tidak termasuk dalam langkah berikutnya.
- **FineAll EX-2 belum lengkap.** RE2 dengan data latih 10% dan 100%, serta seluruh RE1, belum dijalankan. Hasil sejauh ini menunjukkan FineAll tidak lebih baik daripada FineLast, jadi bagian ini tidak diprioritaskan.
- **Encoder hanya dari satu dataset.** Pre-training dari gabungan beberapa dataset belum diuji.
- **Hanya TS2Vec.** TimeMAE dan RUN belum diimplementasikan.
- **Perbedaan perangkat.** Fine-tuning EX-1 berjalan di MPS (Apple M1), sedangkan EX-2 di CPU. Pengaturan yang seharusnya identik bisa berbeda satu kasus uji. Contohnya, encoder acak FineLast di RE1 dengan 100% data latih mendapat 0.938 di EX-1 dan 0.924 di EX-2. Perbedaan ini tidak mengubah kesimpulan. Perbandingan di dalam EX-2 tetap adil karena semua varian dijalankan di perangkat yang sama.

---

## File terkait

| File | Isi |
|---|---|
| `report/ts2vec_ex1_ex2_ppt.md` | Panduan slide |
| `scripts/run_ex1.py`, `scripts/run_ex2.py` | Menjalankan EX-1 dan EX-2 |
| `scripts/diagnose_prior.py` | Uji tebakan dari frekuensi dan anomali dihapus |
| `scripts/paired_test.py` | Uji statistik bootstrap berpasangan |
| `results/ex1/<dataset>_ts2vec/summary.csv` | Skor per run EX-1 |
| `results/ex1/diagnose_prior.csv` | Hasil uji pengecekan EX-1 |
| `results/ex2/<dataset>_ts2vec/summary.csv` | Skor per run EX-2 |
| `results/ex2/<dataset>_ts2vec/predictions.csv` | Rank per kasus uji |
| `results/ex2/<dataset>_ts2vec/paired_test.csv` | Hasil uji statistik |
| `logs/ex1_<dataset>_ts2vec.log`, `logs/ex2_<dataset>_ts2vec_{finelast,fineall}.log` | Log run |
