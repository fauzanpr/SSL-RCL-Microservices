# Daftar TODO dan penanda [belum ada]

Hitungan terakhir (dari `grep` pada seluruh file .tex): 23 `\todo{}`, 54 sel `[belum ada]` di tabel
(tab_ex1: 24, tab_ex2: 8, tab_compare: 22), ditambah 1 penanda di caption tab_ex1.
Dua TODO baru pada putaran ini: pemetaan KPI ke service pada RUN-original dan aturan seri pesimistis. Keduanya
sebelumnya diputuskan sepihak; sekarang ditandai sebagai keputusan yang belum divalidasi beserta konsekuensinya.
Semua TODO di naskah kini berbahasa Indonesia dan memuat konsekuensi bila keputusan salah.

## 0. Keputusan metodologis yang menunggu validasi Anda (bukan sekadar informasi)
| Keputusan | Pilihan sementara di naskah | Konsekuensi bila berbeda |
|---|---|---|
| Aturan trigger KPI RUN-original | belum ditetapkan | data-driven: berbagi informasi dengan heuristik; tetap: butuh pengetahuan topologi |
| Pemetaan KPI ke service RUN-original | skor service = skor PageRank maksimum node-nya | aturan lain mengubah rangking RUN-original dan perbandingan C1 |
| Representasi DLinear untuk RUN-encoder | belum ditetapkan | menentukan apakah encoder merepresentasikan deret atau hasil peramalan |
| Aturan seri | pesimistis (seri dihitung di atas akar masalah) | angka heuristik, RUN-original, dan AC@k lain harus dihitung ulang bila kode lama berbeda |
| Head, loss, dan input mask | belum dikonfirmasi dari kode TS2Vec | C2 dan C3 tidak lagi mengisolasi efek pre-training bila berbeda antar-encoder |

## A. Bisa dilengkapi dari paper atau repositori referensi (belum ada di folder)
| TODO | Lokasi | Sumber |
|---|---|---|
| Optimiser dan batch size TS2Vec (tidak ada di teks utama PDF, ada di appendix) | tab_impl | appendix TS2Vec, atau https://github.com/yuezhihan/ts2vec |
| Keluaran DLinear mana yang menjadi vektor representasi untuk RUN-encoder | 03_methodology | kode RUN, https://github.com/zmlin1998/RUN |
| Konfirmasi KPI yang menjadi node graf RUN (diputuskan: kanal tersedia dari service kandidat) dan perilaku trigger KPI di kode RUN | 03_methodology | kode RUN |

Sudah ditutup pada putaran ini: definisi Avg@5 (diverifikasi dari ART, Bagian 5.1.3), objective TS2Vec/TimeMAE/RUN,
hiperparameter TimeMAE dan RUN di tab_impl, istilah FineLast/FineAll (berasal dari TimeMAE).

## B. Bisa diperoleh dari kode atau data Anda yang sudah ada (tanpa eksperimen baru)
| TODO | Lokasi |
|---|---|
| Arsitektur scoring head, cara mask masuk ke encoder, loss training | 03_methodology |
| Cara subset label 10/25/50% diambil (nested atau tidak); fraksi label di EX-1 | 03_methodology, tab_config |
| Hiperparameter yang dipakai, head, epoch, early stopping, hardware, waktu run | tab_impl |
| Jumlah service per sistem, sampling interval asli, jumlah kasus dengan channel hilang | 04_setup |
| Konfigurasi di balik tabel anomali dihapus (0,91 / 0,9 / 1) dan arti "tebakan acak +-0,06" | tab_anomaly, 05_results |
| Chance level AC@k per dataset (cukup jumlah service tiap kasus uji): 4 sel di tab_compare | tab_compare, 05_results |
| AC@3 heuristik (tanpa pelatihan): 2 sel di tab_compare | tab_compare |
| AC@3 dan Avg@5 TS2Vec, jika skor per kasus tersimpan dari run lama | 05_results |
| Hasil per tipe fault, jika prediksi per kasus tersimpan | 06_discussion |
| Keputusan aturan trigger KPI untuk RUN-original (dipilih di validation saja) | 03_methodology |
| Tes tambahan opsional untuk menguji penjelasan "encoder acak sudah cukup" (keputusan Anda) | 06_discussion |
| Pernyataan competing interests | 07_conclusion |

## C. Harus menunggu eksperimen berikutnya
- tab_ex1: 24 sel (TimeMAE dan RUN-encoder, tiga dataset, FineLast/FineAll, pre-trained dan acak).
- tab_ex2: 8 sel (TimeMAE dan RUN-encoder pada RE2).
- tab_compare: 16 sel (RUN-original, TS2Vec, TimeMAE, RUN-encoder; AC@1 dan AC@3, RE1 dan RE2).
  TS2Vec di tabel ini menunggu konfigurasi terbaik hasil validation, bukan konfigurasi tetap di slide.
- Abstrak, kontribusi akhir, diskusi perbandingan metode, kesimpulan (5 TODO).
- Pelengkap TS2Vec: nilai 10% label di EX-2, FineAll 10% dan 100%, SD untuk grafik EX-2.
