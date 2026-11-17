# Gambar slide progres

Empat PNG 2560 × 1440 (16:9), mengikuti PPT.pdf: logo, warna, tata letak, dan gaya judul. Font lengkap Avenir Next dan Arial digunakan karena embedded font PDF hanya memuat sebagian karakter.

- Slide 1–3: 01_preprocessing_baseline.ipynb.
- Slide 4 RE1/RE2: results/ex1/RE{1,2}_ts2vec/summary.csv dan heuristic_test.csv. Mean ± sample SD dihitung dari 3 seed.
- Hasil EX-1 memakai dataset sumber dan evaluasi yang sama (in-domain), bukan transfer lintas dataset.
- Tanpa pretrain/FineLast memakai encoder acak beku dan head terlatih, sesuai varian scratch.

Render menggunakan Pillow; logo diambil dari PPT.pdf. Tidak menggunakan generasi gambar AI agar tabel dan angka presisi.
