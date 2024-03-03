# LAPORAN PROYEK MACHINE LEARNING - Klasifikasi Kualitas Air 

# Domain Proyek
Air merupakan salah satu kebutuhan yang harus dipenuhi dalam kehidupan manusia[[1](https://ojs.serambimekkah.ac.id/jurnal-biologi/article/view/1592)]. Kehadirannya sangat penting untuk kesehatan dan vitalitas tubuh kita karena tubuh manusia membutuhkan air untuk berfungsi dengan baik[[2](https://www.medichub.ro/reviste-de-specialitate/farmacist-ro/apa-componenta-indispensabila-pentru-sanatatea-si-functionarea-organismului-uman-id-7760-cmsid-62)]. Air berguna untuk pencernaan, penyerapan nutrisi, dan pembuangan sisa[[3](https://scholarworks.uark.edu/cfhndfend/7/)]. Air juga berfungsi untuk mengontrol suhu tubuh, tekanan darah, dan melumasi persendian[[4](https://www.researchgate.net/publication/377303022_EDUKASI_MANFAAT_AIR_MINERAL_PADA_TUBUH_BAGI_ANAK_SEKOLAH_DASAR_SECARA_ONLINE)]. Kekurangan air tubuh bahkan dapat menyebabkan dehidrasi, yang jika tidak diatasi dapat fatal [[5](https://ijhn.ub.ac.id/index.php/ijhn/article/view/114)]. Oleh karena itu, sangat penting bagi setiap orang untuk selalu mengonsumsi jumlah air yang cukup setiap hari agar tubuh tetap sehat.

Di Indonesia, masalah air yang tidak memenuhi standar kualitas masih menjadi perhatian serius. Baik di kota-kota maupun desa, kondisi air tahan di Indonesia semakin buruk [[6](https://iopscience.iop.org/article/10.1088/1755-1315/1190/1/012041)]. Berdasarkan riset dari Kemenkes pada tahun 2020, 74,4% rumah tangga di Indonesia akses air minumnya tercemar oleh bakteri _E.coli_ [[7](https://dataindonesia.id/kesehatan/detail/riset-744-sumber-air-minum-rumah-tangga-ri-tercemar-tinja)]. Air minum yang bersifat basa atau asam dapat mempengaruhi pencernaan dan gangguan lambung, ginjal, dan pembuluh darah [[8](https://media.neliti.com/media/publications/100520-ID-kajian-kualitas-air-dan-penggunaan-sumur.pdf)]. Maka dari itu, memastikan ketersediaan air yang layak untuk diminum harus menjadi prioritas utama pemerintah dan lembaga terkait demi kesejahteraan dan kesehatan seluruh masyarakat Indonesia. 

Untuk memastikan bahwa air aman untuk dikonsumsi, proses penting harus dilakukan berdasarkan berbagai parameter untuk mengukur kualitas air dengan pembuatan model _machine learning_ [[9](https://www.researchgate.net/publication/360650780_Analisis_Komparatif_Algoritme_Machine_Learning_dan_Penanganan_Imbalanced_Data_pada_Klasifikasi_Kualitas_Air_Layak_Minum)] 
_machine learning_ dapat memastikan keamanan konsumsi air karena dapat mendeteksi kualitas air dengan sangat baik dengan mengenali pola dari data historis yang dikumpulkan dari berbagai sumber [[10](https://jurnal.kominfo.go.id/index.php/jpkop/article/view/1752)]. Dalam _machine learning_, banyak algoritma yang dapat digunakan untuk melakukan klasifikasi, seperti KNN, SVM, dan _Random Forest_[[11](https://ieeexplore.ieee.org/document/10134661)]. Dengan memanfaatkan berbagai data untuk setiap variabel yang didapat dan menggunakan algoritma klasifikasi, pembuatan model untuk mengklasifikasikan kualitas air dapat dilakukan[[12](https://mrijet.mrpublishers.com/index.php/mrijet/article/view/10-1-8)].

# Business Understanding

Angka konsumsi air minum yang tidak layak di Indonesia masih tinggi. Maka dari itu, dibutuhkannya pengembangan model _machine learning_ untuk mengklasifikasikan kualitas air minum sebagai sarana untuk membantu dan memastikan apakah air minum yang ingin dikonsumsi layak atau tidak. Salah satu manfaat dari adanya model klasifikasi kualitas air ini adalah model ini dapat digunakan oleh pemerintah atau pihak perusahaan air minum untuk memastikan apakah air yang mereka distibrusikan layak untuk diminum atau tidak. Oleh karena itu, dengan melakukan pengecekan kualitas air minum, konsumen atau masyarakat luas dapat terhindar dari konsumsi air tidak layak minum yang dapat menyebabkan berbagai masalah kesehatan.

### Problem Statements (BELOM)
- Berdasarkan eksplorasi *dataset*, fitur apa saja yang mempengaruhi dalam menentukan estimasi harga rumah?
- Bagaimana mengolah *dataset* agar dapat dibuat model prediksi harga rumah?
- Bagaimanna cara meningkatkan nilai perfoma model prediksi harga rumah?

### Goals (BELOM)
- Mengeksplorasi semua fitur yang tersedia pada *dataset* kemudian membuat melihat korelasi harga dari semua fitur yang sedia untuk melihat faktor apa saja yang paling berpengaruh sampai paling kurang berpengaruh terhadap harga rumah
- Melakukan proses *data wragling* dan *data preparation* terhadap *dataset* agar dapat dibuat model predksi harga rumah
- Melakukan beberapa variasi model untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk prediksi harga rumah


### Solution statements (BELOM)
- Untuk eksplorasi fitur dilakukan Analisis Univariat dan Analisis Multivariat. Analisis Univariat dilakukan untuk mengeksploasi data numerik dan data kategorik. Analisis Multivariat dilakukan untuk melihat hubungan antar fitur. Teknik yang digunakan adalah menggunakan catplot, pairplot, dan heatmap untuk melihat *Correlation Matrix* dari fitur-fitur yang dimiliki.
- Agar didapatkan model prediksi yang baik maka dilakukan proses *Data Wragling* yang meliputi *Data Gathering*, *Data Assessing*, dan *Data Cleaning*.
- Untuk mengetahui perfoma model dilakukan pengecekan performa dengan metrik evaluasi.

# Data Undestanding

_Dataset_ yang digunakan untuk pembangunan model _machine learning_ ini adalah _dataset_ "Water Quality and Potability" yang tersedia di situs web [Kaggle](https://www.kaggle.com/). _Dataset_ tersebut adalah _dataset_ kuantitatif yang berisi kolom-kolom yang dapat menentukan sebuah kualitas air layak diminum atau tidak.

_Dataset_ ini cocok untuk membangun model _supervised learning_, khususnya _binary classification_. Dalam kasus ini adalah untuk mengklasifikasinya sampel sebuah air layak diminum (_Potable_) atau tidak layak diminum (_Not Potable_)

Dataset tersebut dapat diunduh [disini](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability).

Berikut ini adalah informasi lainnya mengenai variabel-variabel yang terdapat di dataset tersebut:

### Variabel-variabel pada _Dataset "Water Quality and Potability"_ adalah sebagai berikut:
- 'pH': Tingkat pH air. 
- 'Hardness': Ukuran kandungan mineral. 
- 'Solids': Total padatan terlarut dalam air. 
- 'Chloramines': Konsentrasi kloramin dalam air. 
- 'Sulfate': Konsentrasi sulfat dalam air. 
- 'Conductivity': Konduktivitas listrik di air. 
- 'Organic_carbon': Kandungan karbon organik dalam air. 
- 'Trihalomethanes': Konsentrasi trihalometan dalam air. 
- 'Turbidity': Tingkat kekeruhan, ukuran kejernihan air. 
- 'Potability': Variabel target. menunjukkan potabilitas air dengan nilai 1 (layak minum) dan 0 (tidak layak minum).

Kemudian, untuk meningkatkan pemahaman atas data terkait, dilakukannya _exploratory data analysis_ dan Visualisasi Data.
- ```python
  dataset.shape
  ```
- ```python
   dataset.keys()
  ```
- ``` dataset.info() ```
- ``` dataset.describe() ```
  
|       |         ph |   Hardness |    Solids |   Chloramines |   Sulfate |   Conductivity |   Organic_carbon |   Trihalomethanes |   Turbidity |   Potability |
|:------|-----------:|-----------:|----------:|--------------:|----------:|---------------:|-----------------:|------------------:|------------:|-------------:|
| count | 2785       |  3276      |  3276     |    3276       | 2495      |      3276      |       3276       |         3114      | 3276        |  3276        |
| mean  |    7.08079 |   196.369  | 22014.1   |       7.12228 |  333.776  |       426.205  |         14.285   |           66.3963 |    3.96679  |     0.39011  |
| std   |    1.59432 |    32.8798 |  8768.57  |       1.58308 |   41.4168 |        80.8241 |          3.30816 |           16.175  |    0.780382 |     0.487849 |
| min   |    0       |    47.432  |   320.943 |       0.352   |  129      |       181.484  |          2.2     |            0.738  |    1.45     |     0        |
| 25%   |    6.09309 |   176.851  | 15666.7   |       6.12742 |  307.699  |       365.734  |         12.0658  |           55.8445 |    3.43971  |     0        |
| 50%   |    7.03675 |   196.968  | 20927.8   |       7.1303  |  333.074  |       421.885  |         14.2183  |           66.6225 |    3.95503  |     0        |
| 75%   |    8.06207 |   216.667  | 27332.8   |       8.11489 |  359.95   |       481.792  |         16.5577  |           77.3375 |    4.50032  |     1        |
| max   |   14       |   323.124  | 61227.2   |      13.127   |  481.031  |       753.343  |         28.3     |          124      |    6.739    |     1        |

- ``` dataset.isnull().sum() ```
- 
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
