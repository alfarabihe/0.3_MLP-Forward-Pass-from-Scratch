# Interpretasi Hasil — Forward Pass MLP dari Nol

Dokumen ini menginterpretasikan output aktual dari `forward_pass_mlp_dari_nol.ipynb`
setelah dijalankan (seed = 42, arsitektur `4 → 8 (tanh) → 2`, batch size = 5).

---

## 1. Inisialisasi Bobot

```
W1 shape: (4, 8)
b1 shape: (8,)
W2 shape: (8, 2)
b2 shape: (2,)
```

Ukuran bobot sudah sesuai rancangan arsitektur: `W1` memetakan 4 fitur input ke 8 neuron
hidden, `W2` memetakan 8 neuron hidden ke 2 neuron output. Karena bobot diinisialisasi
acak dan network **tidak dilatih**, angka-angka pada bobot ini tidak punya makna
"optimal" apa pun — perannya di sini murni sebagai kondisi awal yang identik untuk kedua
skenario (nonlinear vs linear) yang akan dibandingkan.

## 2. Data Batch & Target Acak

```
X shape (5, 4)   -- 5 sampel, masing-masing 4 fitur
Y_target shape (5, 2)  -- 5 target, masing-masing 2 nilai
```

`X` dan `Y_target` juga acak, dan menjadi *fixed reference* yang sama untuk kedua
eksperimen forward pass di bawah. Ini penting secara metodologis: satu-satunya variabel
yang berubah antar eksperimen adalah **ada/tidaknya fungsi aktivasi**, bukan data atau
bobotnya.

## 3. Forward Pass dengan Aktivasi Nonlinear (tanh)

```
Z1 (baris pertama) : [-0.4298  0.8217 -0.6658  0.0631  0.9516  1.5263 -1.2987 -0.0160]
A1 = tanh(Z1)       : [-0.4052  0.6760 -0.5822  0.0630  0.7405  0.9098 -0.8614 -0.0160]
MSE Loss (nonlinear): 0.3584304267056402
```

Perhatikan efek `tanh` secara nilai per nilai:

| Z1 (sebelum tanh) | A1 (sesudah tanh) | Catatan |
|---|---|---|
| 1.5263 | 0.9098 | nilai besar **dijepit mendekati 1** — mulai memasuki daerah *saturasi* |
| -1.2987 | -0.8614 | nilai negatif besar dijepit mendekati -1 |
| 0.0631 | 0.0630 | nilai dekat nol nyaris **tidak berubah** — `tanh` hampir linear di sekitar 0 |

Pola ini adalah bukti visual langsung dari sifat `tanh`: **linear di sekitar nol, tapi
melengkung dan menjenuh (saturasi) menuju ±1 untuk nilai besar**. Efek "melengkungkan"
inilah yang membuat hidden layer tidak bisa lagi diringkas menjadi operasi linear biasa.

## 4. Forward Pass tanpa Aktivasi (Linear Murni)

```
MSE Loss (linear): 0.43686609891080375
```

Dengan bobot dan data yang identik, hanya menghapus `tanh`, hasil output dan loss-nya
**berbeda**: `0.4369` (linear) vs `0.3584` (nonlinear). Perbedaan ini semata-mata berasal
dari absennya nonlinearitas — bukan dari perubahan bobot atau data.

Penting dicatat: nilai loss linear yang lebih tinggi di sini **bukan bukti bahwa
nonlinear "lebih baik" secara umum**. Karena bobot belum dilatih (hanya forward pass
acak), kebetulan saja pada percobaan dengan seed ini nonlinear menghasilkan loss yang
sedikit lebih rendah terhadap target acak. Yang benar-benar bermakna secara teoretis
bukan angka loss-nya, melainkan bagian berikutnya: **bukti bahwa struktur linier itu
runtuh menjadi satu transformasi tunggal**, sedangkan struktur nonlinear tidak.

## 5. Pembuktian: Layer Linear Runtuh Menjadi Satu Transformasi

```
W_eff shape: (4, 2)   b_eff shape: (2,)
Apakah kedua output identik (np.allclose)? True
Selisih absolut maksimum antar keduanya   : 1.67e-16
```

Ini adalah **hasil paling penting** di seluruh notebook. `y_pred_linear` (dari MLP
2-layer tanpa aktivasi) dan `y_pred_single` (dari satu transformasi linear tunggal
`X @ W_eff + b_eff`) menghasilkan angka yang **identik** hingga presisi mesin
(`1.67e-16` — murni galat pembulatan *floating point*, bukan perbedaan matematis nyata).

Artinya secara empiris: **MLP `4→8→2` tanpa fungsi aktivasi setara persis dengan MLP
`4→2` tanpa hidden layer sama sekali.** Menambahkan 8 neuron hidden di tengah tidak
menambah kapasitas model apa pun selama tidak ada nonlinearitas — ia hanya
mereparameterisasi ulang bobot yang sama.

### Bukti tambahan — analisis rank

```
Rank W1  (4x8) : 4
Rank W2  (8x2) : 2
Rank W_eff (4x2) hasil komposisi W1 @ W2 : 2
```

`W1` (rank 4, maksimum untuk ukurannya) dan `W2` (rank 2, maksimum untuk ukurannya)
masing-masing "penuh" secara individual. Tapi begitu dikomposisikan menjadi
`W_eff = W1 @ W2`, rank-nya dibatasi oleh dimensi tersempit dalam rantai perkalian
(`min(4, 8, 2) = 2`). Ini adalah bentuk lain dari *bottleneck* informasi: kapasitas 8
neuron hidden **tidak pernah benar-benar terpakai** dalam rantai transformasi linear —
ia diringkas paksa menjadi ruang berdimensi 2.

## 6. Perbandingan Langsung

```
                         Dengan tanh (nonlinear)   Tanpa aktivasi (linear)
MSE Loss                               0.358430                  0.436866
```

```
Apakah output nonlinear == output linear? False
```

Baris-per-baris, kedua output berbeda cukup jauh untuk beberapa sampel — misalnya sampel
0: `[0.6047, 0.5163]` (nonlinear) vs `[1.0217, 0.7890]` (linear). Perbedaan ini konsisten
dengan efek saturasi `tanh` yang teramati di Bagian 3: nilai pra-aktivasi yang besar
(`Z1` sampel 0 punya elemen sebesar `1.5263`) "dijepit" oleh `tanh`, sehingga informasi
yang diteruskan ke output layer secara sistematis berbeda dari versi linearnya.

## 7. Kesimpulan Interpretatif

1. **Angka MSE loss pada satu percobaan forward pass acak tidak relevan untuk
   menyimpulkan mana yang "lebih baik"** — network belum dilatih, jadi loss ini
   sekadar snapshot dari bobot acak awal.
2. **Yang benar-benar membuktikan pentingnya nonlinearitas adalah kesetaraan matematis
   `y_pred_linear == y_pred_single`** (Bagian 5): tanpa `tanh`, kedalaman network
   (*depth*) tidak menambah apa pun terhadap kapasitas representasinya.
3. **Efek saturasi `tanh` yang teramati pada `Z1` → `A1`** (Bagian 3) adalah gambaran
   nyata dari fungsi yang sama persis dipakai LSTM untuk kandidat *cell state*
   ($\tilde{C}_t = \tanh(\dots)$) — nilai besar dijepit ke ±1, nilai kecil nyaris tidak
   berubah. Perilaku ini adalah dasar mengapa gate LSTM (yang memakai sigmoid, saudara
   dekat `tanh`) bisa bertindak sebagai "saklar" informasi yang halus (*soft switch*),
   bukan sekadar pengali linear.
4. Secara ringkas: hasil eksperimen ini adalah bukti kuantitatif bahwa **kedalaman tanpa
   nonlinearitas adalah ilusi** — sebuah fondasi konseptual yang perlu dipahami sebelum
   mempelajari mengapa setiap gate LSTM wajib dibungkus fungsi aktivasi nonlinear.
