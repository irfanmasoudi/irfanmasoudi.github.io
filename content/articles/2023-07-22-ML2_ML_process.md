---
title: ML2 - Machine Learning Process
tags: Machine Learning, Statistical learning
date: 2023-07-22
description: Pada dasarnya machine learning adalah sebuah proses untung memahami data dengan lebih baik tidak hanya sekadar memodelkan suatu data.
image: 
draft: true
---

### Tentang Data
Pertama dimulai dari data. Mencari, mendapatkan, mengumpulkan lalu menyeleksi data yang benar-benar kita butuhkan, **selection**. Data yang tidak terstruktur, inconsistent, missing some values membutuhkan perlakuan untuk didapatkan data yang benar-benar bisa kita gunakan, **cleaning/pre-processing**. Setelah mendapatkan data yang baik, proses selanjutnya adalah menemukan fitur dari data tersebut, proses yang sering digunakan adalah **data transformation**. Misal, data yang kita miliki adalah data kelahiran, sedangkan yang sebenarnya kita butuhkan adalah data umur. Atau merubah unit ukur raw data menjadi unit ukur tertentu, dengan data scaling atau normalization. Pada proses ini membutuhkan pemahaman dari subjek atau task yang ingin kita selesaikan, hal ini menjadi sangat penting karena merupakan bagian dasar ektraksi fitur. Setelah itu, **mining/learning**, memilih metode yang sesuai dan mengaplikasikannya. **Validation/interpretastion**, mengukur seberapa baik model yang dihasilkan ketika learning. Mengukur sebarapa baik model yang dihasilkan, juga harus bisa dijelaskan sebaik mungkin. Kita harus selalu bersikap kritis, dan tidak serta merta percaya dengan hasil yang baik berdasarkan metrics tertentu. Karena bisa jadi sangat terkait dengan sporious correlation. Itulah kenapa pada proses-proses ini diperlukan tim yang saling melengkapi, expert dimasalah yang ingin kita selesaikan, dan juga expert secara teknis.

### Machine Learning Process
Pada dasarnya machine learning adalah sebuah proses untung memahami data dengan lebih baik tidak hanya sekadar memodelkan suatu data. Contohnya, kita bisa mengaplikasikan linear regression pada suatu data dan ternyata bekerja dengan baik, karena kita tahu bahwa data ini adalah proses linear. Vice-versa, jika tidak berhasil, kita juga harus mampu menjawab kenapa tidak berhasil.

<img class="mx-auto w-4/5 max-w-lg" src="/images/proses.png">

Machine learning proses merupakan sebuah proses yang repetitif, dan tidak selalu terikat pada urutan setiap prosesnya. Misal, dari hasil evaluasi menunjukan bahwa model yang kita dapatkan tidak memiliki hasil yang baik, lalu menyadari bahwa ada kemungkinan preprocessed data yang menyebabkan rusaknya model. Atau kita menyadari bahwa metode yang kita gunakan tidak tepat, maka diperlukan proses training kembali.

#### References
[1] Machine Learning Lecture Prof. Matteo Matteucci\
[2] [The Element of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)\
[3] [An Introduction to Statistical Learning](https://www.statlearning.com/)