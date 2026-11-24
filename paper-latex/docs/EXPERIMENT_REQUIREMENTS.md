# Kebutuhan eksperimen untuk melengkapi artikel

Semua sel `[belum ada]` di tabel adalah eksperimen yang belum dijalankan. Jangan diisi dengan perkiraan.

## A. Prinsip (disetujui): tiga perbandingan terpisah, bukan satu protokol paksa
- C1 performa RCA antar metode (konfigurasi terbaik hasil validation, budget tuning sebanding, termasuk RUN-original dan heuristik).
- C2 efek pre-training/transfer dalam metode (vs encoder acak sendiri, paired bootstrap).
- C3 efek jumlah label dan FineLast/FineAll dalam metode.
- Metrik: AC@1 utama, AC@3 tambahan, Avg@5 (definisi terverifikasi dari ART, Bagian 5.1.3, dan harus diterapkan ke semua metode). Aturan seri pesimistis; laporkan chance level per k.
- Semua metode harus meranking service yang sama per kasus. RUN-original: node graf = kanal tersedia dari service kandidat, skor service = skor PageRank maksimum di antara node-nya. Aturan trigger KPI harus diputuskan dan dipilih lewat validation saja (lihat TODO_LEDGER.md).

## A2. Yang tetap seragam
- Split kasus, 3 seed, preprocessing, dan tensor input [N service, 540, 5] identik dengan TS2Vec yang sudah ada.
- Scoring head yang sama (arsitektur, pooling per service, loss, early stopping pada validation, budget epoch). Dokumentasikan head yang dipakai TS2Vec sekarang, karena slide tidak menjelaskannya.
- Hiperparameter tiap encoder dari paper aslinya (TS2Vec: dim 320 pada paper; TimeMAE: embedding 64, 8 layer, masking 60%, window sub-series dicari {4,8,12,16}, codebook {64..512}; RUN: DLinear, window 32, lr 0.001, batch 128). Tuning hanya pada validation dengan budget yang sama.

## B. TimeMAE (3 seed)
1. EX-1: pre-train dan fine-tune di RE1, RE2, RE3. Pre-trained vs random x FineLast vs FineAll.
2. EX-2: sumber RE1, RE2, RE3 ke target RE1 dan RE2 (sumber != target). Encoder transfer, in-domain, random. Label 10/25/50/100%. FineLast dan FineAll.
3. Paired bootstrap 10.000 kali terhadap random encoder.

## C. RUN
1. RUN-original: pipeline per kasus (forecasting, Granger graph, PageRank) pada test set RE1 dan RE2 sebagai referensi tanpa transfer. Ini unsupervised: tidak ada label dan tidak ada FineLast/FineAll.
2. RUN-encoder: backbone DLinear dengan pre-training kontrastif (SimSiam-style) di bawah scoring head bersama, lalu mengikuti jadwal yang sama dengan B. Ini penyimpangan dari RUN asli dan harus dilaporkan sebagai varian.
3. RUN-original: skor service = peringkat terbaik di antara KPI-nya (RUN merangking KPI); catat cara trigger point dipilih.
4. RUN-encoder selalu disebut varian adaptasi (lihat Bagian 3.4).

## D. Pelengkap TS2Vec (kekurangan data yang sudah ada)
- Angka 10% label pada EX-2 (slide hanya menyebut "semua setara").
- FineAll 10% dan 100% untuk RE1 ke RE2 (bootstrap baru 6 pengaturan).
- Bootstrap untuk RE3 sebagai sumber, dan untuk target RE1 FineAll.
- SD untuk grafik EX-2 dan konfigurasi tepat di balik tabel anomali dihapus (0,91 / 0,9 / 1).
- AC@3 dan Avg@5 (slide hanya mendefinisikan, tidak memberi hasil) bila ingin dilaporkan.

## E. Informasi yang masih harus dilengkapi
Jumlah service per sistem, frekuensi sampling asli, detail scoring head, perangkat keras, durasi pelatihan, pernyataan conflict of interest. Entri ART di references.bib perlu halaman dan DOI dari PDF ART.
