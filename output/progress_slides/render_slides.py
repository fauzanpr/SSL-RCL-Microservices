"""Render empat slide PNG dari hasil lokal; koordinat desain 1280 x 720."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import json
import statistics
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SCALE = 2
INK = '#172f43'
TEAL = '#00aaa9'
GRAY = '#cbd2d3'
MUTED = '#637681'
PALE = '#f0f9f9'
FONT = OUT / 'assets'
BG = Image.open('/tmp/ssl-rcl-template-hi.png').convert('RGB')
BG = BG.resize((2560, 1440))
ImageDraw.Draw(BG).rectangle((0, 270, 2560, 1440), fill='white')

def font(size, kind='body'):
    # PDF fonts are subsets: use complete local faces to avoid missing glyphs.
    if kind in ('title', 'bold'):
        return ImageFont.truetype('/System/Library/Fonts/Avenir Next.ttc', round(size*SCALE), index=0)
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', round(size*SCALE))

def text(x, y, value, size=23, fill=INK, kind='body'):
    d.text((x*SCALE,y*SCALE), value, font=font(size,kind), fill=fill)

def rect(box, fill=None, outline=GRAY, width=1):
    d.rectangle(tuple(round(v*SCALE) for v in box),fill=fill,outline=outline,width=width*SCALE)

def line(points, fill=GRAY, width=1):
    d.line([(int(x*SCALE),int(y*SCALE)) for x,y in points],fill=fill,width=width*SCALE)

def wrapped(x,y,value,width,size=23,fill=INK,kind='body',spacing=1.25):
    words=value.split(); rows=[]; row=''
    for word in words:
        candidate=(row+' '+word).strip()
        if d.textlength(candidate,font=font(size,kind)) > width*SCALE:
            rows.append(row);row=word
        else:row=candidate
    if row:rows.append(row)
    for i,row in enumerate(rows):text(x,y+i*size*spacing,row,size,fill,kind)
    return y+len(rows)*size*spacing

def start(title, subtitle, num):
    global im,d
    im=BG.copy();d=ImageDraw.Draw(im)
    text(72,172,title,40,'#000000','title')
    text(74,230,subtitle,21,MUTED)
    line([(74,660),(1206,660)])
    text(74,674,'Kelompok 3 · Pembelajaran Mesin · Progres eksperimen RCA',15,MUTED)
    text(1156,674,f'{num} / 4',15,MUTED)

def save(name):
    im.save(OUT/name)

start('Karakteristik Dataset RCAEval',
      '733 kasus valid · 3 sistem microservice · input eksperimen saat ini: metrik',1)
xs=[74,339,628,917,1206]
ys=[286,340,400,460,540]
rect((74,286,1206,340),fill=PALE,outline=None)
for x,label in zip(xs,['Karakteristik','RE1','RE2','RE3']):
    if x<1206:text(x+16,297,label,24,TEAL,'bold')
rows=[['Kasus valid','373 / 375','270','90'],
      ['Modalitas tersedia','Metrik','Metrik, log, trace','Metrik, log, trace'],
      ['Jenis fault','CPU, mem, disk,\ndelay, loss','CPU, mem, disk,\ndelay, loss, socket','f1–f5\n(bergantung sistem)']]
for j,row in enumerate(rows):
    y=[354,414,474][j]
    for k,cell in enumerate(row):
        for q,s in enumerate(cell.split('\n')):text(xs[k]+16,y+q*29,s,23)
for y in ys:line([(74,y),(1206,y)])
text(90,557,'Sistem: Online Boutique (OB), Sock Shop (SS), dan Train Ticket (TT).',23)
text(90,592,'RE1: dua kasus dikeluarkan karena waktu injeksi / durasi data tidak valid.',21,MUTED)
text(90,620,'Tantangan: ketersediaan kanal dan jumlah service berbeda antar-dataset.',21,TEAL)
save('01_karakteristik_dataset.png')

start('Preprocessing · Contoh RE1',
      'Prosedur yang sama telah diterapkan pada RE1, RE2, dan RE3.',2)
cards=[(74,'01 · Selaraskan fitur',[
    'Pilih service aplikasi berdasarkan ketersediaan CPU.',
    'K = 5: cpu, mem, latency-50, latency-90, workload.',
    'Kanal tidak tersedia → nilai 0 dan mask = 0.']),
    (460,'02 · Bersihkan & normalisasi',[
    'Samakan grid waktu menjadi 1 detik; forward / backward fill.',
    'Robust z-score dari seluruh periode sebelum injeksi.',
    'Median / MAD; fallback std; batasi z pada −10 sampai +10.']),
    (846,'03 · Bentuk input',[
    'Window: 300 detik sebelum dan 240 detik setelah injeksi.',
    'Tensor x: [N service, 540, 5]. Mask: [N service, 5].',
    'Label: indeks service root cause.'])]
for x,heading,paras in cards:
    rect((x,285,x+360,553))
    text(x+17,303,heading,20,TEAL,'bold')
    y=346
    for p in paras:
        y=wrapped(x+17,y,p,326,21,spacing=1.14)+14
rect((74,575,1206,637),fill=PALE,outline=None)
text(91,583,'Split kasus: train / val / test ≈ 60 / 20 / 20, terstratifikasi per sistem dan root service.',21)
text(91,610,'Pre-training SSL hanya memakai kasus train; val dan test dipisahkan.',21)
save('02_preprocessing.png')

start('Baseline Heuristik',
      'Tanpa training · service paling menyimpang menjadi tebakan pertama root cause',3)
for x,n,title,body in [
    (74,'1','Ambil penyimpangan','Gunakan |z| pada 240 detik setelah injeksi.'),
    (460,'2','Hitung skor service','Rata-rata sepanjang waktu dan kanal yang tersedia.'),
    (846,'3','Urutkan service','Skor terbesar mendapat ranking tertinggi.')]:
    rect((x,285,x+360,432))
    text(x+16,299,n,26,TEAL,'bold')
    text(x+55,305,title,22,TEAL,'bold')
    wrapped(x+17,350,body,325,24)
rect((74,451,1206,517),fill=PALE,outline=None)
text(95,468,'Skor = jumlah |z| pada kanal aktif / (jumlah detik × jumlah kanal aktif)',26,INK,'body')
text(75,538,'Evaluasi',22,TEAL,'bold')
text(75,574,'AC@k: proporsi kasus dengan root cause di k tebakan teratas.',23)
text(75,606,'Avg@5: rata-rata AC@1–AC@5. Waktu injeksi diasumsikan diketahui.',23)
save('03_baseline_heuristik.png')

def read_csv(path):
    with open(path) as f:return list(csv.DictReader(f))

metrics={}
for re in ['RE1','RE2']:
    folder=ROOT/'results'/'ex1'/f'{re}_ts2vec'
    rows=read_csv(folder/'summary.csv')
    groups={}
    for variant in ['pretrain','scratch']:
        for mode in ['FineLast','FineAll']:
            vals=[float(r['AC@1'])*100 for r in rows if r['variant']==variant and r['mode']==mode]
            groups[variant,mode]=(statistics.mean(vals),statistics.stdev(vals))
    heur=read_csv(folder/'heuristic_test.csv')
    baseline=100*sum(int(r['rank'])<=1 for r in heur)/len(heur)
    metrics[re]={'n':len(heur),'baseline':baseline,'groups':groups}

start('Hasil Awal · Model vs Baseline',
      'EX-1 / in-domain · TS2Vec · AC@1 (%) pada split test yang sama',4)
xs=[74,658,932,1206]
rect((74,280,1206,327),fill=PALE,outline=None)
for x,v in zip(xs,['Metode','RE1 · n=75','RE2 · n=45']):
    if x<1206:text(x+14,292,v,21,TEAL,'bold')
entries=[('Heuristik (tanpa training)',None,None),
         ('TS2Vec pretrain · FineLast','pretrain','FineLast'),
         ('TS2Vec pretrain · FineAll','pretrain','FineAll'),
         ('Tanpa pretrain · FineLast','scratch','FineLast'),
         ('Tanpa pretrain · FineAll','scratch','FineAll')]
for i,(label,variant,mode) in enumerate(entries):
    y=339+i*39
    text(88,y,label,22)
    for j,re in enumerate(['RE1','RE2']):
        if variant is None:v=f"{metrics[re]['baseline']:.1f}"
        else:
            mean,std=metrics[re]['groups'][variant,mode];v=f'{mean:.1f} ± {std:.1f}'
        text(xs[j+1]+14,y,v.replace('.',','),22)
    line([(74,y+32),(1206,y+32)])
text(88,547,'Rata-rata ± simpangan baku dari 3 seed (0, 1, 2); FineLast: encoder beku; FineAll: seluruh model dilatih.',17,MUTED)
rect((74,582,1206,641),fill=PALE,outline=None)
text(91,588,'Model melampaui heuristik pada RE1 dan RE2; manfaat pre-training terhadap tanpa pretrain',21)
text(91,613,'masih bervariasi antar-mode. Hasil ini belum menguji transfer lintas dataset.',21)
save('04_hasil_vs_baseline.png')

# Sumber dan status eksplisit agar angka dapat ditelusuri.
(OUT/'README.md').write_text('''# Gambar slide progres\n\nEmpat PNG 2560 × 1440 (16:9), mengikuti PPT.pdf: logo, warna, tata letak, dan gaya judul. Font lengkap Avenir Next dan Arial digunakan karena embedded font PDF hanya memuat sebagian karakter.\n\n- Slide 1–3: 01_preprocessing_baseline.ipynb.\n- Slide 4 RE1/RE2: results/ex1/RE{1,2}_ts2vec/summary.csv dan heuristic_test.csv. Mean ± sample SD dihitung dari 3 seed.\n- Hasil EX-1 memakai dataset sumber dan evaluasi yang sama (in-domain), bukan transfer lintas dataset.\n- Tanpa pretrain/FineLast memakai encoder acak beku dan head terlatih, sesuai varian scratch.\n\nRender menggunakan Pillow; logo diambil dari PPT.pdf. Tidak menggunakan generasi gambar AI agar tabel dan angka presisi.\n''')
with zipfile.ZipFile(OUT/'slide_progres_4_gambar.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.glob('0*.png')):z.write(p,p.name)
thumb=Image.new('RGB',(1280,720),'#e6e8e9')
for i,p in enumerate(sorted(OUT.glob('0*.png'))):
    img=Image.open(p).resize((624,351))
    thumb.paste(img,(8+(i%2)*640,4+(i//2)*360))
thumb.save(OUT/'preview.png')
print('Rendered:',*[p.name for p in sorted(OUT.glob('0*.png'))],sep='\n')
