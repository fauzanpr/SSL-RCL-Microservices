# Menyiapkan project di Overleaf

1. Di Overleaf pilih New Project, lalu template "Springer Lecture Notes in Computer Science (LNCS)". Template itu menyediakan `llncs.cls` dan `splncs04.bst`, yang tidak ada di folder kerja kita. Jangan ganti file class tersebut.
2. Unggah isi arsip `overleaf_upload.zip` ke project (main.tex, references.bib, sections/, tables/, figures/). Hapus `main.tex` bawaan template bila terjadi konflik nama.
3. Menu: Main document = `main.tex`, Compiler = pdfLaTeX, bibliografi dijalankan BibTeX (otomatis).
4. Penanda di PDF: `[TODO: ...]` biru = informasi yang masih dibutuhkan; `[belum ada]` merah = eksperimen belum dijalankan.
5. Sebelum submit, hapus makro `\todo` dan `\pending` (atau ubah ke teks hitam) dan pastikan tidak ada penanda tersisa.

File template asli yang Anda berikan ada di `template_original/samplepaper.tex` (tidak diubah).
Status kompilasi: BELUM diuji. Mohon kompilasi di Overleaf dan kirim log error bila ada.
