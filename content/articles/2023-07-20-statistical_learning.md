---
title: Intro to Statistical Learning
tags: Machine Learning, Statistical learning
date: 2023-07-20
description: Cupcake ipsum dolor. Sit amet pastry cake toffee carrot cake. Cheesecake candy I love dragée cake jelly-o pie. Cheesecake sesame snaps danish lemon drops sesame snaps sugar plum cupcake powder. Cookie sweet wafer. Jelly chocolate cake dragée candy canes halvah.
image: /images/slearning.png
draft: true
---

### Tentang statistical learning

Machine learning memiliki cakupan yang sagat luas. Banyak metode, pendekatan, dan konsep mengenai machine learning yang beririsan dengan statistic. Statistical learning sangat terkait dengan dasar-dasar statistic as a "white-box" method dalam menyelesaikan masalah-masalah machine learning. Pada dasarnya tidak ada satu metode machine learning yang bisa kita gunakan untuk menyelesaikan semua masalah. Maka perlunya kemampuan untuk medeskripsikan model, memiliki intuisi, hipotesa, dan trade-off dari metode yang digunakan menjadi sangat penting.

Melalui pendekatan statistical learning, kita akan belajar bagaimana metode yang akan kita gunakan bekerja, the math behind. Jadi kita tidak hanya bisa mengaplikasikan metode tanpa tahu apa yang sebenarnya terjadi. Bukan karena metode yang kita gunakan bisa bekerja, tetapi karena berdasarkan pendekatan konseptual kenapa metode kita bisa bekerja. Dari data yang ada, lalu membuat hipotesa bahwa metode yang kita gunakan mampu menghasilkan luaran tertentu yang sesuai dengan hipotesis kita. Mudahnya, kita diharapkan memiliki kemampuan untuk mengestimasi akan seperti apa hasilnya.

### Machine learning overview

Definisi Machine Learning sangat susah didefinisikan secara pasti. Karena didalamnya banyak metode dan pendekatan dari berbagai macam disiplin pengetahuan, dari statistics, computer science, pattern recognition, ataupun knowledge discovery. Jadi, machine learning lebih cocok masuk sebagai sebuah permasalahan daripada sebuah disiplin ilmu pengetahuan. Yakni masalah tentang data, pattern, dan prediksi.

Tetapi kita bisa berangkat dari definisi Tom Mitchell - 1997 tentang machine learning:

> A computer program is said to learn from experience E with respect
> to some class of task T and a performance measure P, if its 
> performance at tasks in T, as measured by P, improves because of
> experience E.

Dimana $T$: Regression/Classification/Clustering etc, $E$: Data, $P$: Errors/Loss. 

Kita akan membedakan machine learning berdasarkan kategori learning paradigm-nya. Asumsikan kita memiliki $E$, adalah data,

$input=x_{1},x_{2},x_{3},\dots,x_{N}$

$x$ bisa berbentuk skalar ataupun vektor. i.e. umur karyawan $x_{1} = 28$ atau $x_{1} = \begin{bmatrix} 160\\ 50 \end{bmatrix}$ adalah tinggi dan berat karyawan. Selain itu kita juga memiliki data $target=y_{1},y_{2},y_{3},\dots,x_{N}$. i.e. Gaji karyawan dalam juta IDR $t_{1}=3.5$ atau dalam bentuk kategori $t_{1}=kaya$. Ketika tugas kita memprediksi target $t_{k}$ berdasarkan input baru $x_{k}$ dengan $P$ yang baik, maka kita sebut ini sebagai **Supervised Learning**.

Jika kita hanya memiliki $input$ saja tanpa $target$, maka kita kategorikan paradigma ini sebagai **Unsupervised Learning**, mencari regularities atau kesamaan pattern dari representasi data, untuk reasoning atau prediksi.

Paradigma lain yang mungkin akan kita pelajari pada series tulisan lain adalah **Reinceforcement Learning**. Dengan melakukan aksi $a_{1},a_{2},a_{3},\dots,a_{N}$ yang mempengaruhi sistem, dimana setiap aksi berkaitan dengan reward $r_{1},r_{2},r_{3},\dots,r_{N}$ yang kemudian kita gunakan sebagai parameter untuk memaksimalkan reward selanjutnya.

### Supervised learning: Classification

$X\to Handcrafted\;Features \to Learned\;Classifier \to Y$, dimana $X$ adalah input, dan $Y$ adalah target atau output. Handcrafted feature adalah karakteristik atau measurable property yang kita gunakan untuk mengidentifikasi suatu data. Learned classifier adalah hasil dari proses training atau biasa kita sebut dengan istilah Model. $X\to Model \to Y$, jadi Model adalah suatu $fungsi$ yang akan kita gunakan untuk memprediksi output $1$ atau $0$ **(binary classification)** berdasarkan input $x_{k}$. Atau memprediski target $kucing$, $anjing$, atau $bebek$ **(multiclass classification)** berdasarkan input $x_{k}$.

### Supervised learning: Regression
Sama halnya dengan classification, $X\to Handcrafted\;Features \to Learned\;Regressor \to Y$,  $X\to Model \to Y$ bedanya, $Y$ adalah **numerical value**. Baik classification atau resgression, model adalah suatu fungsi $f(X)$. Sehingga learning suatu model sematically equivalent dengan learning suatu fungsi. Jika task kita adalah linear regression, maka kita learning fungsi linear. Jika polinomial regression, maka learning fungsi polinomial. Logistic regression, learning logistic function dst. Dimana pemilihan class function, kita yang akan menentukan. Itulah kenapa kita harus memiliki kemampuan untuk menentukan hipotesa dengan tepat, bahwa masalah ini bisa kita selesaikan dengan funsi linier, quadratic, atau polinomial, konsep ini kita kenal dengan istilah **Inductive Hypothesis**.

### Unsupervised learning: Clustering
Misalkan kita memiliki data point $x=\begin{bmatrix} berat \\ tinggi \\ umur \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \\ 1 \end{bmatrix}$ Data point dapat kita representasikan ke dalam space pada titik koordinat $v$. Kita bisa sebut vektor karena $\lVert x \rVert$ adalah magnitude dari origin $(0,0,0)$ ke titik $v$ seperti pada gambar.

  <img class="mx-auto w-4/5 max-w-xs" src="/images/3dvector1.png">

Jadi ketika kita melihat suatu data, kita dapat merepresentasikannya kedalam bentuk space. Dengan kata lain, kita bisa melihat setiap axis pada space adalah sebuah fitur dari data. Kombinasi dan transformasi axis dapat kita gunakan untuk mengalisa suatu data. Jika $X\in{\Reals^p} \to Model \to Y\in \cal{P}(X)$ dimana $X$ memiliki p-dimensional space dan output $Y$ adalah powerset dari $X$. Maka clustering adalah pengelompokan atau partitioning $X$ berdasarkan kesamaan dari fiturnya.
