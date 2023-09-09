---
title: ML3 - Algebra mini-course
tags: Machine Learning, Statistical learning
date: 2023-08-02
description: Sebelum masuk ke materi utama, saya rasa kita perlu belajar materi-materi algebra yang berkaitan dengan machine learning. Saya coba merangkum beberapa topik yang menurut saya akan berguna untuk memahami materi dan konsep machine learning.
image: 
draft: true
---

### Selayang pandang
Sebelum masuk ke materi utama, saya rasa kita perlu belajar materi-materi algebra yang berkaitan dengan machine learning. Saya coba merangkum beberapa topik yang menurut saya akan berguna untuk memahami materi dan konsep machine learning.

### Vectors
Vector adalah list dari beberapa bilangan yang kita sebut dengan element. Dimensi dari vektor adalah rows x columns (e.g 5x1, column vector). Column vector, $v = \begin{bmatrix} 1\\ -2\\ 0\\ 9.3\\-.01\end{bmatrix}$ dan transpose dari column vector adalah row vector, $v^{T} = \begin{bmatrix} 1 & -2 & 0 & 9.3 & -.01 \end{bmatrix}$. 

### Matrices
Matriks adalah susunan bilangan berdasarkan rows dan column, dua dimensi.
$A=[a_{ij}]_{m \times n}=\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \dots & \dots & \dots & \dots \\ a_{m1} & \dots & \dots & a_{mn} \end{bmatrix}$, m adalah row dan n adalah column. 

### Interpretasi vector
<img class="mx-auto w-4/5 max-w-xs" src="/images/vector1.png">

Misal kita memiliki $v = \begin{bmatrix} v_{1}\\ v_{2} \end{bmatrix}$, magnitude dari matrix v, $\lVert v \rVert = \sqrt{v_{1}^{2}+v_{2}^{2}}$ adalah ecludean distance dari origin ke titik v. Orientation dari v, $\theta = tan^{-1} \lparen \dfrac{v_{1}}{v_{2}} \rparen$. Jika  $\lVert v \rVert = 1$, maka v adalah unit vector, atau biasa disebut dengan versor atau unit norm, yang dapat kita gunakan sebagai definisi arah dari vektor. $\overrightarrow{v}=\dfrac{v}{\lVert v \rVert} = \begin{bmatrix} v_{1}/\lVert v \rVert\\ v_{2}/\lVert v \rVert \end{bmatrix}$. 

### Vector dot dan cross product
Vector dot product sering kita sebut dengan inner product atau scalar product. Dot/inner product dari vector berbentuk skalar.

<img class="mx-auto w-4/5 max-w-xs" src="/images/vector2.png">

$v = \begin{bmatrix} v_{1}\\ v_{2} \end{bmatrix}, w = \begin{bmatrix} w_{1}\\ w_{2} \end{bmatrix}$. Inner product $v.w = \begin{bmatrix} v_{1}\\ v_{2} \end{bmatrix}.\begin{bmatrix} w_{1}\\ w_{2} \end{bmatrix} = v_{1}w_{1} + v_{2}w_{2} = \lVert v \rVert . \lVert w \rVert \cos \alpha$. Dot product kita gunakan untuk mengetahui apakah vector v dan w orthogonal/tegak lurus atau tidak. Jika $v \; \bot \; w $ tegak lurus maka $v.w = \lVert v \rVert . \lVert w \rVert \cos \alpha = 0$.

Vector cross product dari vector berbentuk vektor. $u=v \times w$, magnitude $\lVert u \rVert=\lVert v \times w \rVert = \lVert v \rVert . \lVert w \rVert \sin \alpha$. Orientasi $u \; \bot \; v = u . v \Rightarrow \lparen v \times w \rparen . v = 0, \; u \; \bot \; w = u . w \Rightarrow \lparen v \times w \rparen . w = 0$ 
<img class="mx-auto w-4/5 max-w-xs" src="/images/vector3.png">
Jika $v \parallel w, \; v \times w = \lVert v \rVert . \lVert w \rVert \sin \alpha = 0$ maka vector v dan w paralel.

### Matrix Properties
Jika $A=[a_{ij}]_{m \times n}$ maka transpose dari matrix A, $A^{T}=[a_{ji}]_{n \times m}$. Hal lain yang juga penting kita ketahui, $(A + B)^{T}=A^{T}+B^{T}, \: (AB)^{T}=B^{T}+A^{T}$. Jika $A=A^{T}$ maka $A$ adalah symmetric matrix. Commutativity dan associativity matrix, $AB \not = BA, \; A(BC) = (AB)C$

### Matrix Trace dan Determinant
Determinant dari suatu matrix dapat dihitung jika matrix $A$ adalah square matrix. $A=[a_{ij}]_{n \times n};\; det(A)=\displaystyle\sum_{j=1}^n a_{ij}A_{ij}; \;i=1,\dots,n$. Sedangkan Trace matrix $A$, $A=[a_{ij}]_{n \times n};\;tr[A]=\displaystyle\sum_{j=1}^n a_{jj}$ 

### Matrix Inverse
Matrix inverse adalah proprerti matrix yang hanya mungkin dimiliki oleh square matrix. $A=[a_{ij}]_{n \times n}$. $AA^{-1}=A^{-1}A=I$ dimana $I$ adalah identity matrix. $A^{-1}=\begin{bmatrix}a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}^{-1} = \dfrac{1}{det(A)}\begin{bmatrix}a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix},\;$ dimana $det(A) \not = 0$. Jadi jika $det(A)=0$ maka matrix $A$ tidak memiliki inverse.

### Linear Independence
Vector $x_{i} \in \Reals^{n}$ dikatakan linearly independent jika vektor $x_{i}$ tidak dapat dihasilkan dari linear combination dari vektor lain. Misal, $x_{1}=\begin{bmatrix}1\\2\end{bmatrix}, \; x_{2}=\begin{bmatrix}-2\\-4\end{bmatrix}$ jika $c=\begin{bmatrix}1\\1/2\end{bmatrix}$ maka $c_{1}x_{1}+c_{2}x_{2}=0$. Sehingga $x_{1}$ dan $x_{2}$ **not** linearly independent. Jika $A=\begin{bmatrix}x_{1}&x_{2}\end{bmatrix}=\begin{bmatrix}1&-2\\2&-4\end{bmatrix}$ maka $det(A) = 0$. Sehingga jika kita ingin memiliki $A^{-1}$ maka, setiap column dari matrix $A$ harus liniearly independent.

### Span
$span(x_{1},x_{2}, \dots ,x_{i})=\lbrace c_{1}x_{1}+c_{2}x_{2}+ \dots + c_{i}x_{i} \; \vert \; c_{1}, c{2}, \dots ,c_{i} \in \Reals \rbrace$. Setiap titik pada space, misal $w$ dapat kita representasikan dengan $w=c_{1}v_{1}+c_{2}v_{2}, \; w = \begin{bmatrix}w_{1}\\w_{2}\end{bmatrix}=w_{1}v_{1}+w_{2}v_{2}$

<img class="mx-auto w-4/5 max-w-xs" src="/images/vector5.png">

dan $v_{1}$, $v_{2}$ linearly independent, sehingga dapat kita gunakan sebagai basis vector. Karena $v_{1} \bot v_{2}$ dan memiliki magnitude 1 unit vector, maka juga disebut dengan orthonormal basis.

### Eigenvalue dan Eigenvector

Eigenvalue $\lambda$ dan Eigenvector $u$ memenuhi $Ax=\lambda u$ dimana $A$ adalah square matrix. Mengkalikan $u$ dengan $A$ dengan skala $\lambda$.

<img class="mx-auto w-4/5 max-w-xs" src="/images/vector4.png">

 $Ax=\lambda u = (A- \lambda I)u=0$ dimana hanya memiliki solusi jika $det(A- \lambda I)=0$.

#### References
[1] Machine Learning Lecture Prof. Matteo Matteucci\
[2] [Basic Linear Algebra](http://www.ekof.bg.ac.rs/wp-content/uploads/2016/09/Ponavljanje-matematike-Wayne-Winston-Operations-Research-Applications-and-Algorithms-4-edition.pdf)
