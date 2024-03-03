# LAPORAN PROYEK MACHINE LEARNING - Klasifikasi Kualitas Air 

# Domain Proyek
Air merupakan salah satu kebutuhan yang harus dipenuhi dalam kehidupan manusia[[1](https://ojs.serambimekkah.ac.id/jurnal-biologi/article/view/1592)]. Kehadirannya sangat penting untuk kesehatan dan vitalitas tubuh kita karena tubuh manusia membutuhkan air untuk berfungsi dengan baik[[2](https://www.medichub.ro/reviste-de-specialitate/farmacist-ro/apa-componenta-indispensabila-pentru-sanatatea-si-functionarea-organismului-uman-id-7760-cmsid-62)]. Air berguna untuk pencernaan, penyerapan nutrisi, dan pembuangan sisa[[3](https://scholarworks.uark.edu/cfhndfend/7/)]. Air juga berfungsi untuk mengontrol suhu tubuh, tekanan darah, dan melumasi persendian[[4](https://www.researchgate.net/publication/377303022_EDUKASI_MANFAAT_AIR_MINERAL_PADA_TUBUH_BAGI_ANAK_SEKOLAH_DASAR_SECARA_ONLINE)]. Kekurangan air tubuh bahkan dapat menyebabkan dehidrasi, yang jika tidak diatasi dapat fatal [[5](https://ijhn.ub.ac.id/index.php/ijhn/article/view/114)]. Oleh karena itu, sangat penting bagi setiap orang untuk selalu mengonsumsi jumlah air yang cukup setiap hari agar tubuh tetap sehat.

Di Indonesia, masalah air yang tidak memenuhi standar kualitas masih menjadi perhatian serius. Baik di kota-kota maupun desa, kondisi air tahan di Indonesia semakin buruk [[6](https://iopscience.iop.org/article/10.1088/1755-1315/1190/1/012041)]. Berdasarkan riset dari Kemenkes pada tahun 2020, 74,4% rumah tangga di Indonesia akses air minumnya tercemar oleh bakteri E.coli [[7](https://dataindonesia.id/kesehatan/detail/riset-744-sumber-air-minum-rumah-tangga-ri-tercemar-tinja)]. Air minum yang bersifat basa atau asam dapat mempengaruhi pencernaan dan gangguan lambung, ginjal, dan pembuluh darah [[8](https://media.neliti.com/media/publications/100520-ID-kajian-kualitas-air-dan-penggunaan-sumur.pdf)]. Maka dari itu, memastikan ketersediaan air yang layak untuk diminum harus menjadi prioritas utama pemerintah dan lembaga terkait demi kesejahteraan dan kesehatan seluruh masyarakat Indonesia. 

Untuk memastikan bahwa air aman untuk dikonsumsi, proses penting harus dilakukan berdasarkan berbagai parameter untuk mengukur kualitas air dengan pembuatan model  Machine Learning [[9](https://www.researchgate.net/publication/360650780_Analisis_Komparatif_Algoritme_Machine_Learning_dan_Penanganan_Imbalanced_Data_pada_Klasifikasi_Kualitas_Air_Layak_Minum)] 
Machine learning dapat memastikan keamanan konsumsi air karena dapat mendeteksi kualitas air dengan sangat baik dengan mengenali pola dari data historis yang dikumpulkan dari berbagai sumber [[10](https://jurnal.kominfo.go.id/index.php/jpkop/article/view/1752)]. Dalam Machine Learning, banyak algoritma yang dapat digunakan untuk melakukan klasifikasi, seperti KNN, SVM, dan Random Forest[[11](https://ieeexplore.ieee.org/document/10134661)]. Dengan memanfaatkan berbagai data untuk setiap variabel yang didapat dan menggunakan algoritma klasifikasi, pembuatan model untuk meng-klasifikasi kualitas air dapat dilakukan[[12](https://mrijet.mrpublishers.com/index.php/mrijet/article/view/10-1-8)].

# Business Understanding

Angka konsumsi air minum yang tidak layak di Indonesia masih tinggi. Maka dari itu, dibutuhkannya pengembangan model machine learning untuk mengklasifikasikan kualitas air minum sebagai sarana untuk membantu dan memastikan apakah air minum yang ingin dikonsumsi layak atau tidak. Salah satu manfaat dari adanya model klasifikasi kualitas air ini adalah model ini dapat digunakan oleh pemerintah atau pihak perusahaan air minum untuk memastikan apakah air yang mereka distibrusikan layak untuk diminum atau tidak. Oleh karena itu, dengan melakukan pengecekan kualitas air minum, konsumen atau masyarakat luas dapat terhindar dari konsumsi air tidak layak minum yang dapat menyebabkan berbagai masalah kesehatan.

### Problem Statements
- Berdasarkan eksplorasi *dataset*, fitur apa saja yang mempengaruhi dalam menentukan estimasi harga rumah?
- Bagaimana mengolah *dataset* agar dapat dibuat model prediksi harga rumah?
- Bagaimanna cara meningkatkan nilai perfoma model prediksi harga rumah?

### Goals
- Mengeksplorasi semua fitur yang tersedia pada *dataset* kemudian membuat melihat korelasi harga dari semua fitur yang sedia untuk melihat faktor apa saja yang paling berpengaruh sampai paling kurang berpengaruh terhadap harga rumah
- Melakukan proses *data wragling* dan *data preparation* terhadap *dataset* agar dapat dibuat model predksi harga rumah
- Melakukan beberapa variasi model untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk prediksi harga rumah


### Solution statements
- Untuk eksplorasi fitur dilakukan Analisis Univariat dan Analisis Multivariat. Analisis Univariat dilakukan untuk mengeksploasi data numerik dan data kategorik. Analisis Multivariat dilakukan untuk melihat hubungan antar fitur. Teknik yang digunakan adalah menggunakan catplot, pairplot, dan heatmap untuk melihat *Correlation Matrix* dari fitur-fitur yang dimiliki.
- Agar didapatkan model prediksi yang baik maka dilakukan proses *Data Wragling* yang meliputi *Data Gathering*, *Data Assessing*, dan *Data Cleaning*.
- Untuk mengetahui perfoma model dilakukan pengecekan performa dengan metrik evaluasi.

# Data Undestanding
# Data Preparation
# Modelling
# Evaluation
## Referensi (NANTI TANYA)

[1] Z. Ab, Ismail Efendy, D. Syamsul, and I. wati, “FAKTOR YANG BERHUBUNGAN TINGKAT KONSUMSI AIR BERSIH PADA RUMAH TANGGA DI KECAMATAN PEUDADA KABUPATEN BIREUN,” vol. 7, no. 2, Nov. 2019, doi: https://doi.org/10.32672/jbe.v7i2.1592.
‌

[1] Ab, Z., Ismail Efendy, Syamsul, D., & wati, I. (2019). FAKTOR YANG BERHUBUNGAN TINGKAT KONSUMSI AIR BERSIH PADA RUMAH TANGGA DI KECAMATAN PEUDADA KABUPATEN BIREUN. 7(2). https://doi.org/10.32672/jbe.v7i2.1592

‌[2]

[3]

[4]

[5]

[6]

[7]

[8]

[9]

[10]

[11]

[12]
