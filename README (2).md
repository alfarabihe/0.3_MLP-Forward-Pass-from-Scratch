# Forward Pass MLP dari Nol

Implementasi forward pass Multi-Layer Perceptron (MLP) 1 hidden layer **dari nol**
menggunakan `numpy` murni (tanpa framework deep learning), untuk memahami secara
matematis dan eksperimental mengapa fungsi aktivasi nonlinear (`tanh`, `sigmoid`)
membuat neural network lebih ekspresif dibanding sekadar rangkaian transformasi linear.

Project ini adalah fondasi konseptual sebelum mempelajari LSTM — setiap *gate* pada LSTM
(*forget*, *input*, *output*) selalu dibungkus `sigmoid`, dan kandidat *cell state*
selalu dibungkus `tanh`. Memahami kenapa nonlinearitas itu wajib adalah tujuan utama
repo ini.

## Isi Repo

| File | Deskripsi |
|---|---|
| `forward_pass_mlp_dari_nol.ipynb` | Notebook utama: implementasi, eksperimen, dan pembuktian |

## Apa yang Dibangun

- MLP dengan arsitektur **4 input → 8 hidden (aktivasi `tanh`) → 2 output**
- Hanya **forward pass** — tidak ada training/backpropagation, bobot diinisialisasi acak
- Dua skenario dibandingkan dengan bobot & data yang identik:
  1. Forward pass **dengan** aktivasi `tanh` di hidden layer
  2. Forward pass **tanpa** aktivasi (semua layer murni linear)
- Pembuktian matematis + verifikasi numerik (`np.allclose`) bahwa MLP linear bertumpuk
  runtuh (*collapse*) menjadi satu transformasi linear tunggal
- Analisis rank matriks bobot sebagai bukti tambahan hilangnya kapasitas model saat
  tidak ada nonlinearitas

## Hasil Utama

Dengan bobot, input, dan target yang **identik** di kedua skenario:

| | Dengan `tanh` (nonlinear) | Tanpa aktivasi (linear) |
|---|---|---|
| MSE Loss | 0.3584 | 0.4369 |

Bukti inti dari project ini:

```
Apakah output MLP linear 2-layer == output satu transformasi linear tunggal? True
Selisih absolut maksimum: 1.67e-16   (murni galat pembulatan floating point)
```

Artinya: **MLP `4→8→2` tanpa fungsi aktivasi secara matematis identik dengan MLP `4→2`
tanpa hidden layer sama sekali.** Kedalaman (*depth*) tanpa nonlinearitas tidak menambah
kapasitas representasi apa pun — hanya nonlinearitas (seperti `tanh`) yang membuat setiap
layer tambahan benar-benar berarti.

Interpretasi lengkap dari seluruh output notebook tersedia di [`INTERPRETASI.md`](./INTERPRETASI.md).

## Cara Menjalankan

Kebutuhan: Python 3.x, `numpy`, Jupyter (atau Jupyter-compatible editor seperti
VS Code/JupyterLab).

```bash
pip install numpy notebook
jupyter notebook forward_pass_mlp_dari_nol.ipynb
```

Jalankan seluruh cell secara berurutan (*Run All*). Notebook bersifat deterministik
(menggunakan `random seed` tetap) sehingga hasil yang didapat akan selalu sama setiap
kali dijalankan ulang.

## Struktur Notebook

1. Pengantar & tujuan project
2. Teori dasar forward pass MLP dan MSE loss
3. Inisialisasi bobot acak
4. Data batch & target acak
5. Forward pass dengan aktivasi `tanh`
6. Forward pass tanpa aktivasi (linear murni)
7. Pembuktian matematis + verifikasi numerik: layer linear runtuh menjadi satu transformasi tunggal
8. Perbandingan langsung nonlinear vs linear
9. Kesimpulan tertulis & kaitannya dengan gate LSTM

## Relevansi ke LSTM

Setiap *gate* pada LSTM ($f_t$, $i_t$, $o_t$) dibungkus **sigmoid** agar bernilai di
rentang $(0,1)$ dan bisa berfungsi sebagai *filter*/saklar informasi. Kandidat *cell
state* $\tilde{C}_t$ dibungkus **`tanh`** agar bernilai di $(-1,1)$. Tanpa nonlinearitas
ini, seluruh mekanisme *gating* LSTM akan runtuh menjadi kombinasi linear biasa — persis
seperti yang dibuktikan pada MLP sederhana di repo ini.
