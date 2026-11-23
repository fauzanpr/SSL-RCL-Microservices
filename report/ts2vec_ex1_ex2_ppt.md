# Panduan slide: TS2Vec, EX-1 dan EX-2

Sepuluh slide, kira-kira 10–12 menit. Setiap slide berisi isi utama, saran visual, dan satu kalimat untuk disampaikan secara lisan. Penjelasan detail setiap slide, dengan urutan yang sama, ada di `report/ex2_ts2vec.md`.

---

## Bagian A: Latar

### Slide 1: Masalah dan tujuan

**Isi**
- Saat terjadi gangguan di sistem microservice, kita ingin tahu **service mana yang menjadi sumber masalah** (root-cause).
- Data yang dipakai adalah metrik per service: CPU, memori, latensi, dan beban. Dataset yang dipakai: RCAEval RE1, RE2, dan RE3, masing-masing dengan tiga sistem (ob, ss, tt).
- Pertanyaan riset: **apakah pre-training tanpa label (self-supervised) membantu menemukan root-cause?**

**Ucapkan:** "Kita ingin tahu apakah model yang lebih dulu belajar pola metrik tanpa label bisa lebih jago menunjuk service yang bermasalah."

---

### Slide 2: Cara kerja dan pembanding

**Isi**
1. **Pre-training (tanpa label):** encoder TS2Vec belajar pola dari metrik normal dan metrik saat gangguan.
2. **Fine-tuning (dengan label):** model dilatih untuk memberi skor ke setiap service.
3. **Uji:** apakah service dengan skor tertinggi memang root-cause-nya? Metrik ini disebut **AC@1**.

Ada dua pembanding:
- **Encoder acak**: tanpa pre-training. Pre-training dianggap berguna hanya kalau mengalahkan ini.
- **Heuristik**: pilih service dengan lonjakan metrik terbesar.

**Visual:** alur tiga kotak: `Pre-training` → `Fine-tuning` → `Uji AC@1`. Bisa juga memakai potongan `report/Pipeline.png`.

**Ucapkan:** "Pembanding terpenting adalah encoder acak, karena dengan itu kita bisa memisahkan efek pre-training dari efek fine-tuning."

---

## Bagian B: EX-1, dilatih dan diuji di dataset yang sama

### Slide 3: Hasil EX-1

**Isi:** pre-training dan pengujian dilakukan di dataset yang sama, misalnya RE1 → RE1.

AC@1 pada data uji, rata-rata dari 3 seed:

| Dataset | Heuristik | Encoder acak | Pre-training TS2Vec |
|---|---|---|---|
| RE1 (75 kasus uji) | 0.81 | 0.93–0.94 | 0.92–0.95 |
| RE2 (45 kasus uji) | 0.80 | 0.86–0.87 | 0.86–0.89 |
| RE3 (18 kasus uji) | 0.67 | 1.00 | 1.00 |

Rentang angka menunjukkan dua cara melatih: encoder dibekukan (FineLast) atau ikut dilatih (FineAll).

**Visual:** diagram batang per dataset, dengan tiga batang: heuristik, acak, dan pre-training.

**Ucapkan:** "Model jauh lebih baik dari heuristik. Tapi encoder acak juga sama baiknya, jadi pre-training belum terlihat membantu."

---

### Slide 4: Apakah model benar-benar membaca anomali?

**Isi**
- Uji pengecekan: bagian metrik setelah gangguan diganti dengan data normal dari kasus yang sama. Jika model memang membaca anomali, skornya seharusnya jatuh.
- Hasilnya, AC@1 jatuh ke tingkat tebakan acak:

| Dataset | Data normal | Anomali dihapus | Tebakan acak |
|---|---|---|---|
| RE1 | 0.91 | 0.03 | ±0.06 |
| RE2 | 0.91 | 0.09 | ±0.06 |
| RE3 | 1.00 | 0.00 | ±0.06 |

- **Kesimpulan EX-1:** model membaca anomali dan tidak sekadar menghafal service. Tapi **pre-training ≈ encoder acak** di dataset yang sama.

**Ucapkan:** "Modelnya benar, tapi manfaat pre-training belum terbukti. Karena itu kita lanjut ke EX-2."

---

## Bagian C: EX-2, encoder dipindah ke dataset lain

### Slide 5: Pertanyaan dan cara EX-2

**Isi**
- Pertanyaan: **apakah encoder dari dataset lain bisa membantu?** Misalnya pre-training di RE1, lalu dipakai untuk RE2.
- Encoder yang dibandingkan:
  - transfer (dari dataset lain),
  - in-domain (dari dataset tujuan sendiri, sama seperti EX-1),
  - acak.
- Jumlah data latih divariasikan (10%, 25%, 50%, 100%) untuk meniru kondisi label sedikit.
- RE3 tidak dipakai sebagai dataset tujuan karena semua model sudah sempurna di sana.

**Visual:** dua panah.
- EX-1: RE2 → RE2
- EX-2: RE1 → RE2 dan RE3 → RE2

**Ucapkan:** "Kalau pre-training tidak membantu di dataset yang sama, mungkin manfaatnya muncul saat dipindah."

---

### Slide 6: Hasil utama EX-2 (RE2)

**Isi**
- Encoder dari **RE1** adalah yang terbaik di RE2: sekitar **0.90–0.91** vs encoder acak **0.84–0.86** (dengan data latih 25% ke atas).
- Selisihnya sekitar 2–3 kasus dari 45 kasus uji.
- Di RE1 sebagai dataset tujuan, semua encoder setara, sama seperti EX-1.

**Visual:** diagram batang AC@1 RE2 per jumlah data latih (25%, 50%, 100%). Isinya empat batang per kelompok (transfer RE1, transfer RE3, in-domain, acak), plus garis heuristik di 0.80.

**Ucapkan:** "Ada peningkatan, tapi kecil."

---

### Slide 7: Apakah bukan kebetulan?

**Isi**
- Data uji kecil (45 kasus), jadi diuji dengan **bootstrap berpasangan**: kasus uji diambil ulang 10.000 kali.
- Transfer RE1 → RE2 **bermakna di 4 dari 5 pengaturan** (data latih ≥ 25%).
- Di RE1 sebagai dataset tujuan: **tidak ada yang bermakna**.

**Visual:** grafik selisih vs encoder acak, dengan rentang 95% (*error bar*) dan garis di 0.

**Ucapkan:** "Kalau rentangnya tidak menyentuh nol, selisihnya bukan kebetulan."

---

### Slide 8: Yang tidak sesuai harapan

**Isi**
- **Label sangat sedikit (10%)**: semua encoder setara. Pre-training tidak membantu di kondisi yang paling diharapkan.
- **Encoder dari RE3** tidak membantu. **RE1 sebagai tujuan** juga tidak terbantu oleh encoder mana pun.
- **Melatih ulang encoder (FineAll) tidak lebih baik** daripada membekukannya, karena model cepat overfit.

**Ucapkan:** "Manfaatnya hanya muncul di satu pasangan dataset, bukan pola umum."

---

## Bagian D: Penutup

### Slide 9: Kesimpulan

**Isi**
- Model berbasis TS2Vec **mengalahkan heuristik** (+0.05 sampai +0.33 AC@1) dan terbukti membaca anomali.
- Tapi keunggulan itu berasal dari **fine-tuning**, bukan dari pre-training. Encoder acak sama baiknya (EX-1).
- Pre-training baru terlihat membantu saat **dipindah dari RE1 ke RE2** (EX-2). Efeknya kecil dan tidak berlaku umum.

**Ucapkan:** "SSL belum terbukti sebagai kunci. Ada satu sinyal positif yang perlu diperkuat."

---

### Slide 10: Langkah berikutnya

**Isi**
1. Ulangi EX-1 dan EX-2 dengan **TimeMAE** dan **RUN**.
2. Tambahkan varian PCA dan dataset ART.

**Visual:** `report/figures/langkah_berikutnya.png`.

**Ucapkan:** "Selanjutnya kami akan mengulang kedua eksperimen dengan dua metode self-supervised lain, lalu menambahkan varian PCA dan dataset ART."

---

## Catatan untuk penyaji

- Fokus ke **AC@1**. AC@3 dan AC@5 hampir selalu tinggi karena kandidat root-cause hanya 2–5 service per sistem.
- Selisih +0.33 di slide 9 berasal dari RE3 (1.00 vs heuristik 0.67). Di RE1 dan RE2, selisihnya sekitar +0.05 sampai +0.13.
- Kalau ditanya "kenapa RE3 tidak dipakai di EX-2?": semua model, termasuk encoder acak, sudah mencapai AC@1 = 1.0, jadi tidak ada yang bisa dibandingkan. Data ujinya juga hanya 18 kasus.
- Kalau ditanya "kenapa RE1 → RE2 berhasil, tapi RE2 → RE1 tidak?": kemungkinan karena RE1 punya data latih terbanyak (223 kasus), sehingga encoder-nya belajar pola yang lebih umum. Dugaan ini **belum diuji**.
- Kalau ditanya soal kebocoran data: pre-training hanya memakai data latih. Uji "anomali dihapus" di slide 4 juga menunjukkan model tidak sekadar menghafal service.
