# LAPORAN PROYEK MACHINE LEARNING - Klasifikasi Kualitas Air 

# Domain Proyek
Air merupakan salah satu kebutuhan yang harus dipenuhi dalam kehidupan manusia[[1](https://ojs.serambimekkah.ac.id/jurnal-biologi/article/view/1592)]. Kehadirannya sangat penting untuk kesehatan dan vitalitas tubuh kita karena tubuh manusia membutuhkan air untuk berfungsi dengan baik[[2](https://www.medichub.ro/reviste-de-specialitate/farmacist-ro/apa-componenta-indispensabila-pentru-sanatatea-si-functionarea-organismului-uman-id-7760-cmsid-62)]. Air berguna untuk pencernaan, penyerapan nutrisi, dan pembuangan sisa[[3](https://scholarworks.uark.edu/cfhndfend/7/)]. Air juga berfungsi untuk mengontrol suhu tubuh, tekanan darah, dan melumasi persendian[[4](https://www.researchgate.net/publication/377303022_EDUKASI_MANFAAT_AIR_MINERAL_PADA_TUBUH_BAGI_ANAK_SEKOLAH_DASAR_SECARA_ONLINE)]. Kekurangan air tubuh bahkan dapat menyebabkan dehidrasi, yang jika tidak diatasi dapat fatal [[5](https://ijhn.ub.ac.id/index.php/ijhn/article/view/114)]. Oleh karena itu, sangat penting bagi setiap orang untuk selalu mengonsumsi jumlah air yang cukup setiap hari agar tubuh tetap sehat.

Di Indonesia, masalah air yang tidak memenuhi standar kualitas masih menjadi perhatian serius. Baik di kota-kota maupun desa, kondisi air tahan di Indonesia semakin buruk [[6](https://iopscience.iop.org/article/10.1088/1755-1315/1190/1/012041)]. Berdasarkan riset dari Kemenkes pada tahun 2020, 74,4% rumah tangga di Indonesia akses air minumnya tercemar oleh bakteri _E.coli_ [[7](https://dataindonesia.id/kesehatan/detail/riset-744-sumber-air-minum-rumah-tangga-ri-tercemar-tinja)]. Air minum yang bersifat basa atau asam dapat mempengaruhi pencernaan dan gangguan lambung, ginjal, dan pembuluh darah [[8](https://media.neliti.com/media/publications/100520-ID-kajian-kualitas-air-dan-penggunaan-sumur.pdf)]. Maka dari itu, memastikan ketersediaan air yang layak untuk diminum harus menjadi prioritas utama pemerintah dan lembaga terkait demi kesejahteraan dan kesehatan seluruh masyarakat Indonesia. 

Untuk memastikan bahwa air aman untuk dikonsumsi, proses penting harus dilakukan berdasarkan berbagai parameter untuk mengukur kualitas air dengan pembuatan model _machine learning_ [[9](https://www.researchgate.net/publication/360650780_Analisis_Komparatif_Algoritme_Machine_Learning_dan_Penanganan_Imbalanced_Data_pada_Klasifikasi_Kualitas_Air_Layak_Minum)] 
_machine learning_ dapat memastikan keamanan konsumsi air karena dapat mendeteksi kualitas air dengan sangat baik dengan mengenali pola dari data historis yang dikumpulkan dari berbagai sumber [[10](https://jurnal.kominfo.go.id/index.php/jpkop/article/view/1752)]. Dalam _machine learning_, banyak algoritma yang dapat digunakan untuk melakukan klasifikasi, seperti KNN, SVM, dan _Random Forest_[[11](https://ieeexplore.ieee.org/document/10134661)]. Dengan memanfaatkan berbagai data untuk setiap variabel yang didapat dan menggunakan algoritma klasifikasi, pembuatan model untuk mengklasifikasikan kualitas air dapat dilakukan[[12](https://mrijet.mrpublishers.com/index.php/mrijet/article/view/10-1-8)].

# Business Understanding

Angka konsumsi air minum yang tidak layak di Indonesia masih tinggi. Maka dari itu, dibutuhkannya pengembangan model _machine learning_ untuk mengklasifikasikan kualitas air minum sebagai sarana untuk membantu dan memastikan apakah air minum yang ingin dikonsumsi layak atau tidak. Salah satu manfaat dari adanya model klasifikasi kualitas air ini adalah model ini dapat digunakan oleh pemerintah atau pihak perusahaan air minum untuk memastikan apakah air yang mereka distibrusikan layak untuk diminum atau tidak. Oleh karena itu, dengan melakukan pengecekan kualitas air minum, konsumen atau masyarakat luas dapat terhindar dari konsumsi air tidak layak minum yang dapat menyebabkan berbagai masalah kesehatan.

### Problem Statements
- Berdasarkan eksplorasi terhadap dataset, fitur-fitur apa saja yang dapat menentukan atau memberi pengaruh terhadap klasifikasi layak atau tidaknya air minum?
- Bagaimana memproses dataset agar dapat digunakan untuk pembuatan model machine learning klasifikasi kualitas air minum?
- Bagaimana cara medapatkan model klasifikasi kuaitas air minum dengan performa terbaik?


### Goals
- Melakukan eksplorasi semua fitur-fitur yang terdapat pada dataset dan melihat fitur-fitur mana saja yang memiliki pengaruh besar atau memiliki korelasi tinggi terhadap label klasifikasi air.
- Melakukan data preparation untuk mempersiapkan model untuk proses training.
- Melakukan proses training dengan baseline model dari berbagai algoritma dan menggunakan baseline model dengan performa terbaik untuk melakukan tuning hyperparameter.


### Solution statements
- Untuk melakukan eksplorasi fitur dilakukan analisis univariat dan multivariat untuk menemukan hubungan antar fitur baik data numerik maupun data kategorikal. Kemudian, menggunakan barchart, heatmap, dan correlation matrix untuk medapatkan informasi lebih lanjut
- Untuk mendapatkan data yang bersih untuk diproses ke tahap modelling, dilakukannya proses data preparation yang terdiri dari data cleaning, train test split, dan data transformation. Kebersihan data dapat mempengaruhi performa model yang akan dibuat.
- Untuk mendapatkan model dengan performa terbaik, digunakan 3 algoritma sebagai baseline model, yaitu KNN, SVM, dan Random Forest. Kemudian, untuk dapat mengetahui baseline model mana yang memiliki performa terbaik dapat dilakukan evaluasi menggunakan Confusion Matrix (Accuracy, Precision, Recall, F1 Score) yang juga divisualisasikan. Selanjutnya, model yang terpilih akan dibantu dengan grid search untuk menemukan hyperparameter yang memiliki performa terbaik. Terakhir, model tersebut dilakukan evaluasi menggunakan Confusion Matrix (Accuracy, Precision, Recall, F1 Score).

  
# Data Undestanding

_Dataset_ yang digunakan untuk pembangunan model _machine learning_ ini adalah _dataset_ "Water Quality and Potability" yang tersedia di situs web [Kaggle](https://www.kaggle.com/). _Dataset_ tersebut adalah _dataset_ kuantitatif yang berisi kolom-kolom yang dapat menentukan sebuah kualitas air layak diminum atau tidak. _Dataset_ ini memiliki 3276 baris dan 10 kolom data.

_Dataset_ ini cocok untuk membangun model _supervised learning_, khususnya _binary classification_. Dalam kasus ini adalah untuk mengklasifikasinya sampel sebuah air layak diminum (_Potable_) atau tidak layak diminum (_Not Potable_)

_Dataset_ tersebut dapat diunduh [disini](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability).

Berikut ini adalah informasi lainnya mengenai variabel-variabel yang terdapat di dataset tersebut:

### Variabel-variabel pada _Dataset "Water Quality and Potability"_ adalah sebagai berikut:
- ```pH```: Tingkat pH air. 
- ```Hardness```: Ukuran kandungan mineral. 
- ```Solids```: Total padatan terlarut dalam air. 
- ```Chloramines```: Konsentrasi kloramin dalam air. 
- ```Sulfate```: Konsentrasi sulfat dalam air. 
- ```Conductivity```: Konduktivitas listrik di air. 
- ```Organic_carbon```: Kandungan karbon organik dalam air. 
- ```Trihalomethanes```: Konsentrasi trihalometan dalam air. 
- ```Turbidity```: Tingkat kekeruhan, ukuran kejernihan air. 
- ```Potability```: Variabel target. menunjukkan potabilitas air dengan nilai 1 (layak minum) dan 0 (tidak layak minum).

Kemudian, untuk meningkatkan pemahaman atas data terkait, dilakukannya _exploratory data analysis_ dan Visualisasi Data.

**_Exploratory Data Analysis_**
- ```python
  dataset.shape
  ```
  Kode diatas memiliki output:
  ```python
  (3276, 10)
  ```

  Berdasarkan _output_ tersebut, didapatkan informasi bahwa dataset ini memiliki **3276 baris** dan **10 kolom** data sesuai dengan dengan keterangan yang tertera diatas. Pada bagian ini, belum dapat diketahui **nama** dari **kolom-kolom** yang ada.
- ```python
   dataset.keys()
  ```
  Kode diatas memiliki output:
  ```python
  Index(['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'],
      dtype='object')
  ```

  Berdasarkan _output_ tersebut, didapatkan informasi lebih lanjut bahwa dataset ini memiliki 10 kolom sesuai dengan keterangan yang tertera diatas. Pada bagian ini, belum dapat diketahui **jumlah** dan **tipe data** dari tiap kolom.
- ```python
   dataset.info()
  ```
  Kode diatas memiliki output:
  ```python
  RangeIndex: 3276 entries, 0 to 3275
  Data columns (total 10 columns):
   #   Column           Non-Null Count  Dtype  
  ---  ------           --------------  -----  
   0   ph               2785 non-null   float64
   1   Hardness         3276 non-null   float64
   2   Solids           3276 non-null   float64
   3   Chloramines      3276 non-null   float64
   4   Sulfate          2495 non-null   float64
   5   Conductivity     3276 non-null   float64
   6   Organic_carbon   3276 non-null   float64
   7   Trihalomethanes  3114 non-null   float64
   8   Turbidity        3276 non-null   float64
   9   Potability       3276 non-null   int64  
  dtypes: float64(9), int64(1)
  ```

   Berdasarkan _output_ tersebut, didapatkan informasi mengenai jumlah data dan tipe data dari setiap kolom yang ada. Ada beberapa kolom yang tidak miliki jumlah data sesuai dengan total baris, yaitu 3276 baris. Hal ini mengindikasikan adanya _missing value_. Kemudian, hanya 1 kolom yang bertipe ```int64```, yaitu kolom ```'Potability'```. Kolom lainnya bertipe ```float64```.
- ```python
   dataset.describe()
  ```
  Kode diatas memiliki output:
  
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
  
   Berdasarkan _output_ tersebut, didapatkan informasi mengenai statistika deskriptif dari _dataset_ yang digunakan. Berikut ini adalah keterangan untuk setiap bagian:
   - ```count``` : Jumlah data dari sebuah kolom
   - ```mean``` : Rata-rata dari sebuah kolom
   - ```std``` : Standar deviasi dari sebuah kolom
   - ```min``` : Nilai terendah pada sebuah kolom
   - ```25%``` : Nilai kuartil pertama (Q1) dari sebuah kolom
   - ```50%``` : Nilai kuartil kedua (Q2) atau median atau nilai tengah dari sebuah kolom
   - ```75%``` : Nilai kuartil ketiha (Q3) dari sebuah kolom
   - ```max``` : Nilai tertinggi pada sebuah kolom   
   
- ```python
  dataset.isnull().sum()
  ```
  
  ```python
  
  ph                 491
  Hardness             0
  Solids               0
  Chloramines          0
  Sulfate            781
  Conductivity         0
  Organic_carbon       0
  Trihalomethanes    162
  Turbidity            0
  Potability           0
  dtype: int64
  ```
  Berdasarkan _output_ tersebut, ditemukan beberapa _missing value_ pada beberapa variabel, yaitu ```pH```, ```Sulfate```, ```Trihalomethanes```. _Missing value_ perlu ditangani agar tidak berdampak buruk kepada model yang akan dibuat.

**Visualisasi Data**
  - Univariate Analysis
    
    ![Univariate-2](https://github.com/ensiklopedical/Water-Quality-Classification/assets/115972304/7d02cde4-cc22-49f7-ba8f-34bca9b09f3b)
    <div align="center">Gambar 1a - Univariate Analysis Categorical Column</div>
    
    Berdasarkan ``` Gambar 1a ``` , terlihat bahwa ```Potability``` memiliki dua _unique_ value, yaitu '1' yang menyatakan air layak minum dan '0' yang menyatakan air tidak layak minum. Namun, terlihat juga bahwa adanya _imbalance data_ atau ketidakseimbangan data. nilai '0' memiliki baris data hingga nyaris 2000 baris data, sedangkan nilai '1' hanya memiliki sekitar 1250 baris data. Berangkat dari informasi ini, perlu dilakukan penyeimbangan agar tidak terjadi bias pada model _machine learning_ yang akan dibangun.

    ![Univariate](https://github.com/ensiklopedical/Water-Quality-Classification/assets/115972304/629564a6-dbbe-4199-984d-4af686952318)
    <div align="center">Gambar 1b - Univariate Analysis Numeric Column</div>
    
    Berdasarkan ```Gambar 1b```, gambar ini menampilkan setiap kolom numerik yang ada pada dataset, seperti ```pH```, ```Hardness```, ```Solids```, ```Chrolamines```, ```Sulfate```, ```Conductivity```, ```Organic_carbon```, ```Trihalomethanes```, ```Turbidity```. Dari semua kolom yang ditampilkan, hanya kolom ```Solids``` dan ```Conductivity``` yang memiliki skewness ke arah kiri. Berikut adalah informasi singkat yang didapatkan dari visualisasi diatas:

    - ```pH```: Tingkat pH berkisar antara 0 hingga 14, dengan sebagian besar sampel memiliki pH sekitar 7, yang merupakan netral. 
    - ```Hardness```: Kesadahan air bervariasi, dengan sejumlah besar sampel menunjukkan tingkat kesadahan sekitar 200. 
    - ```Solids```: Terdapat kisaran total padatan terlarut dalam sampel, dengan konsentrasi puncak mendekati 20.000.
    - ```Chloramines```: Kadar kloramin dalam sampel mencapai puncaknya mendekati 7 atau 8. 
    - ```Sulfate```: Konsentrasi sulfat mencapai puncaknya sekitar 300.
    - ```Conductivity```: Konduktivitas sampel menunjukkan puncak pusat mendekati 400.
    - ```Organic_carbon```: Nilai paling umum untuk kandungan karbon organik adalah sekitar 14 hingga 15.
    - ```Trihalomethanes```: Frekuensi kadar trihalomethane paling tinggi sekitar 65 hingga 70.
    - ```Turbidity```: Tingkat kekeruhan mencapai puncaknya sekitar 3,5. 

  - Multivariate Analysis

    ![Multivariate-1](https://github.com/ensiklopedical/Water-Quality-Classification/assets/115972304/de724feb-5c4b-4339-b6f8-9ed31affcf4c)
    <div align="center">Gambar 2a - Multivariate Analysis Categorical Column - Every Numeric Column</div>

    ![Multivariate-2](https://github.com/ensiklopedical/Water-Quality-Classification/assets/115972304/bac7770e-d08d-464a-b56d-0e2ee19f2761)
    <div align="center">Gambar 2b - Multivariate Analysis Categorical Column - Numeric Column based on Potability</div>

    Berdasarkan gambar ```Gambar 2a``` dan ```Gambar 2b```, dapat terlihat nyaris semua variabel berkumpul di tengah dan tidak menunjukkan karakteristik atau pola khusus terhadap variabel label, yaitu ```'Potability'```. Bahkan, pada ```Gambar 2b``` sekalipun yang sudah di kategorikan berdasarkan ```0``` dan ```1``` (ditandai dengan warna oren dan biru) masih tidak terlihat karakterisik atau pola untuk _value_ pada label tertentu. Kejadian ini mengindikasikan rendahnya korelasi antar fitur, bahkan dengan variabel label sekalipun.

    
  - Correlation

    ![Correlation](https://github.com/ensiklopedical/Water-Quality-Classification/assets/115972304/160fd61a-cddd-4c72-8f53-8def354bd3dd)
    <div align="center">Gambar 3a - Multivariate Analysis Categorical Column - Numeric Column based on Potability</div>

    Berdasarkan ```Gambar 3a```, terlihat bahwa kolom ```pH```, ```Conductivity```, ```Trihalomethanes```, ```Turbidity``` memiliki skor korelasi yang paling kecil terhadap label. Kolom yang semacam ini baiknya di-drop saja untuk meringankan beban komputasi dan mengurangi dimensi dari dataset yang akan digunakan dalam pelatihan model.
    
  - Missing Value

    ![Missing Value](https://github.com/ensiklopedical/Water-Quality-Classification/assets/115972304/7302ab07-57ef-4147-a3b6-ed75f87561a5)
    <div align="center">Gambar 4a - Multivariate Analysis Categorical Column - Numeric Column based on Potability</div>

    Berdasarkan ```Gambar 4a```, terlihat jelas bahwa memang terdapat banyak kekosongan data atau _missing value_ pada ketiga kolom, yaitu ```Sulfate```, ```ph```, dan ```Trihalomethanes```. Kondisi ini perlu tindakan lebih lanjut agar tidak mempengaruhi performa model.
    
# Data Preparation
Data Preparation adalah proses pembersihan, transformasi, dan pengorganisasian data mentah ke dalam format yang dapat dipahami oleh algoritma pembelajaran mesin. Berikut ini adalah **urutan** langkah-langkah Data Preparation yang dilakukan beserta penjelasan dan alasannya:

- Data Cleaning

  ```python

  ```
  
  Data cleaning adalah adalah langkah penting dalam proses Machine Learning karena melibatkan identifikasi dan penghapusan data yang hilang, duplikat, atau tidak relevan yang terdapat pada dataset. Proses ini memiliki berbagai langkah yang perlu dilakukan supaya dataset siap digunakan untuk pembangunan model Machine Learning.
    
  **Alasan**: Data Cleaning diperlukan agar data yang digunakan akurat, konsisten, dan bebas kesalahan, karena data yang salah atau tidak konsisten dapat berdampak negatif terhadap performa model Machine Learning
    - Detection and Removal Duplicates
      
      Data duplikat adalah baris data yang sama persis untuk setiap variabel yang ada. Dataset yang digunakan perlu diperiksa juga apakah dataset memiliki data yang sama atau data duplikat. Jika ada, maka data tersebut harus ditangani dengan menghapus data duplikat tersebut.

      **Alasan**: Data duplikat perlu didektesi dan dihapus karena jika dibiarkan pada dataset dapat membuat model Anda memiliki bias, sehingga menyebabkan overfitting. Dengan kata lain, model memiliki performa akurasi yang baik pada data pelatihan, tetapi buruk pada data baru. Menghapus data duplikat dapat membantu memastikan bahwa model Anda dapat menemukan pola yang ada lebih baik lagi.

      Berikut ini adalah proses pendeteksian dan penghapusan data duplikatnya:
      ```python
      # Cek baris duplikat dalam dataset
      duplicates = dataset.duplicated()
      
      # Hitung jumlah baris duplikat
      duplicate_count = duplicates.sum()
      
      # Cetak jumlah baris duplikat
      print(f"Number of duplicate rows: {duplicate_count}")

      ```

      Berikut ini adalah hasilnya:

      ```python
        Number of duplicate rows: 0
      ```

      Berdasarkan hasil tersebut, tidak ditemukan adanya data duplikat, maka tidak ada juga proses penghapusannya.
      
      
    - Dropping Column with Low Correlation
      
      Pada bagian ini adalah proses penghapusan fitur-fitur yang memiliki korelasi rendah terhadap variabel target dari dataset. Langkah ini diambil berdasarkan asumsi bahwa fitur dengan korelasi rendah tidak memberikan kontribusi signifikan terhadap prediksi yang dibuat oleh model.
 
      **Alasan**: Tahapan ini perlu dilakukan karena fitur dengan korelasi rendah terhadap variabel target cenderung tidak memberikan informasi yang berguna untuk prediksi dan dapat menambahkan kebisingan yang tidak perlu ke dalam model. Dengan menghilangkan fitur-fitur ini, kita dapat mengurangi kompleksitas model, yang dapat membantu dalam mencegah overfitting dan mempercepat waktu pelatihan. Selain itu, model yang lebih sederhana dengan fitur yang lebih sedikit lebih mudah untuk diinterpretasikan, yang memungkinkan kita untuk lebih memahami bagaimana fitur-fitur tersebut mempengaruhi variabel target. 

      Berikut ini adalah proses penghapusan kolom dengan korelasi yang rendah:
      ```python
       # Mendefinisikan daftar fitur dengan korelasi rendah terhadap variabel target
      low_corr = ['ph', 'Trihalomethanes', 'Turbidity', 'Conductivity']
      
      # Menghapus fitur-fitur tersebut dari dataset
      # Axis=1 menunjukkan bahwa operasi penghapusan dilakukan pada kolom (fitur)
      dataset = dataset.drop(low_corr, axis=1)

      ```
      Berikut ini adalah tampilan dataframe setelah penghapusan beberapa kolom:
      
      | Hardness   | Solids     | Chloramines | Sulfate  | Organic Carbon | Potability |
      |------------|------------|-------------|----------|----------------|------------|
      | 204.890455 | 20791.31898| 7.300212    | 368.516441 | 10.379783      | 0          |
      | 129.422921 | 18630.057858| 6.635246   | NaN      | 15.180013      | 0          |
      | 224.236259 | 19909.541732| 9.275884   | NaN      | 16.868637      | 0          |
      | 214.373394 | 22018.417441| 8.059332   | 356.886136| 18.436524      | 0          |

      Penghapusan kolom dengan korelasi rendah sudah berhasil dilakukan.

    - Handle Missing Value
      
      Missing Value terjadi ketika variabel atau barus tertentu kekurangan titik data, sehingga menghasilkan informasi yang tidak lengkap. Nilai yang hilang dapat ditangani dengan berbagai cara seperti imputasi (mengisi nilai yang hilang dengan mean, median, modus, dll), atau penghapusan (menghilangkan baris atau kolom yang nilai hilang)
 
      **Alasan**: Missing Value perlu ditangani karena jika dibiarkan dapat berpengaruh ke rendahnya akurasi model yang akan dibuat. Maka dari itu, penting untuk mengatasi missing value secara efisien untuk mendapatkan model Machine Learning yang baik juga.
 
      Berikut ini adalah kode untuk mencari tahu kolom mana saja dan berapa jumlah missing value-nya:
      ```python
       dataset.isnull().sum()
      ```
 
      Berikut ini adalah output-nya:
      ```python
      Hardness            0
      Solids              0
      Chloramines         0
      Sulfate           781
      Organic_carbon      0
      Potability          0
      dtype: int64
      ```

      Berikut ini kode untuk menghapus baris data yang memiliki missing value:
      ```python
       dataset.dropna(inplace =True)
      ```

      Penanganan missing value sudah berhasil dilakukan.
      
    - Outliers Detection and Removal
      
      Outliers adalah titik data yang menyimpang secara signifikan dari data-data lainnya yang ada. Outliers bisa saja terdapat di hampir semua variabel. Maka dari itu, penting untuk dideteksi dan dihapus jika ada.
    
      **Alasan**:Outliers perlu dideteksi dan dihapus karena jika dibiarkan dapat merusak hasil analisis statistik pada kumpulan data sehingga menghasilkan performa model yang kurang baik. Selain itu, Mendeteksi dan menghapus outlier dapat membantu meningkatkan performa model Machine Learning menjadi lebih baik.
 
      
      ![Boxplots](https://github.com/ensiklopedical/Water-Quality-Classification/assets/115972304/ff0bc57e-003e-4701-b296-b920174168e151)
      <div align="center">Gambar 5a - Boxplots Outlier</div>

      Berikut ini adalah kode untuk menghapus outliers yang ada pada dataframe:
      ```python
      # Assuming 'df' is your DataFrame
      Q1 = dataset.quantile(0.25)
      Q3 = dataset.quantile(0.75)
      IQR = Q3 - Q1
      
      # Define bounds for what is considered an outlier
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      
      # Remove outliers
      dataset = dataset[~((dataset < lower_bound) | (dataset > upper_bound)).any(axis=1)]
      ```
      
      Penghapusan outliers sudah berhasil dilakukan.

    - Imbalance Data
      
      Imbalance data adalah kondisi di mana kelas atau kategori dalam dataset tidak diwakili secara merata, dengan satu kelas mendominasi yang lain. Jika hal ini dibiarkan hingga proses pelatihan model dapat mengakibatkan bias pada model. Hal ini bisa diatasi dengan oversampling atau undersampling.

      **Alasan**: Hal ini dapat menjadi masalah adalah karena imbalance data dapat menyebabkan model bias terhadap kelas mayoritas (lebih banyak) dan menghasilkan performa yang buruk pada kelas minoritas lebih sedikit)

      Berikut ini adalah untuk memeriksa ada berapa baris data untuk masing-masing kelas pada kolom ```'Potability'```:
      ```python
        count_0 = dataset[dataset['Potability'] == 0].shape[0]
        count_1 = dataset[dataset['Potability'] == 1].shape[0]
        print("Jumlah baris data yang berinilai '0' ada sebanyak: " + str(count_0))
        print("Jumlah baris data yang berinilai '0' ada sebanyak: " + str(count_1))
      ```
 
      Berikut ini adalah hasilnya:
      ```python
        Jumlah baris data yang berinilai '0' ada sebanyak: 1431
        Jumlah baris data yang berinilai '1' ada sebanyak: 883
      ```

      Dalam hal ini, perlunya dilakukannya undersampling terhadap kelas ```'0'``` agar menyesuaikan jumlah baris datanya dengan kelas ```'1'```
 
      Berikut ini adalah bagian untuk melakukan proses undersampling:
      ```python
        dataset = dataset.groupby('Potability').apply(lambda x: x.sample(min(len(x), min(count_0, count_1)))).reset_index(drop=True)
      ```
     
      Berikut ini adalah untuk memeriksa ada berapa baris data untuk masing-masing kelas pada kolom ```'Potability'``` setelah dilakukan undersampling:
      ```python
        count_0 = dataset[dataset['Potability'] == 0].shape[0]
        count_1 = dataset[dataset['Potability'] == 1].shape[0]
        print("Jumlah baris data yang berinilai '0' ada sebanyak: " + str(count_0))
        print("Jumlah baris data yang berinilai '0' ada sebanyak: " + str(count_1))
      ```

      Berikut ini adalah hasil setelah dilakukannya undesampling:
      ```python
      Jumlah baris data yang berinilai '0' ada sebanyak: 883
      Jumlah baris data yang berinilai '1' ada sebanyak: 883    
      ```

     Proses penyeimbangan dataset sudah berhasil dilakukan.
  
- Train Test Split
  
  Train Test Split adalah metode yang digunakan untuk membagi dataset menjadi dua bagian: satu untuk melatih model (training set) dan satu lagi untuk menguji model (testing set). Biasanya, data dibagi dengan proporsi tertentu, misalnya 80% untuk training dan 20% untuk testing.

  **Alasan**: Proses ini dilakukan agar dapat mengevaluasi kinerja model secara objektif. Dengan memisahkan data uji, kita dapat mengukur seberapa baik model memprediksi data baru yang tidak pernah dilihat sebelumnya, yang merupakan indikator penting dari kemampuan generalisasi model.

  Berikut adalah bagian untuk membagi dataset menjadi train set dan test set:
  ```python
  X = dataset.drop(["Potability"], axis =1)
  y = dataset["Potability"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
  ```

  Berikut ini adalah bagian untuk memeriksa ada berapa baris data untuk train dan test pada dataframe variabel non-label:
  ```pyhton
  train_count = X_train.shape
  test_count = X_test.shape
  print("Dataset training memiliki data sebanyak " + str(train_count[0]) + " baris")
  print("Dataset test memiliki data sebanyak " + str(test_count[0]) + " baris")
  ```
  
  Berikut ini adalah hasilnya:
  ```python
  Dataset training memiliki data sebanyak 1412 baris
  Dataset test memiliki data sebanyak 354 baris
  ```

  Berikut ini adalah bagian untuk memeriksa ada berapa baris data untuk train dan test pada dataframe variabel label:
  ```python
  train_count_label = y_train.shape
  test_count_label = y_test.shape
  print("Dataset label training memiliki data sebanyak " + str(train_count_label[0]) + " baris")
  print("Dataset label test memiliki data sebanyak " + str(test_count_label[0]) + " baris")
  ```

   Berikut ini adalah hasilnya:
  ```pyhton
  Dataset label training memiliki data sebanyak 1412 baris
  Dataset label test memiliki data sebanyak 354 baris
  ```

  Train Test Split sudah berhasil dilakukan.
  
- Data Transformation
  
  Data Transformation adalah proses mengubah data dari satu format atau struktur ke format atau struktur lainnya. Proses ini biasanya dari format sistem sumber menjadi yang dibutuhkan oleh sistem tujuan. Data Transformation dapat dilakukan dengan berbagai cara, seperti mengubah satuan ukuran data, mengubah distribusi data, atau mengubah bentuk data.
    
    **Alasan**: Data Tranformasi perlu dilakukan karena dapat meningkatkan efisiensi dan meningkatkan kualitas data yang digunakan dalam pembuatan model Machine Learning.
    - Standardization
 
      Standardisasi adalah proses mengubah data menjadi format yang lebih seragam dan dapat dibandingkan. Ini biasanya melibatkan pengurangan rata-rata (mean) dan pembagian dengan simpangan baku (standard deviation) untuk setiap fitur, sehingga fitur tersebut akan memiliki rata-rata nol dan varians satu.

      **Alasan**: Standardisasi perlu dilakukan karena banyak algoritma machine learning yang berperforma lebih baik jika fitur-fitur berada pada skala yang sama. Standardisasi membantu dalam hal ini dengan memastikan bahwa setiap fitur berkontribusi secara proporsional ke hasil akhir dan menghindari bias terhadap fitur dengan skala yang lebih besar.

      Berikut ini adalah penerapan standardisasinya:
      ```python
      X_train[:] = scaler.fit_transform(X_train[:])
      ```

      Setelah dilakukannya standardisasinya, dapat kita cek hasilnya perubahannya dengan melihat mean dan standar deviasinya
      
      Berikut ini adalah keadaan setelah dilakukan standardisasi:
      |       |   Hardness |   Solids |   Chloramines |   Sulfate |   Organic_carbon |
      |:------|-----------:|---------:|--------------:|----------:|-----------------:|
      | count |   354      | 354      |      354      |  354      |         354      |
      | mean  |     0      |   0      |        0      |    0      |          -0      |
      | std   |     1.0014 |   1.0014 |        1.0014 |    1.0014 |           1.0014 |
      | min   |    -2.5788 |  -2.1347 |       -2.614  |   -2.6012 |          -2.8722 |
      | 25%   |    -0.5844 |  -0.7659 |       -0.6995 |   -0.6544 |          -0.6918 |
      | 50%   |     0.0714 |  -0.1126 |       -0.0591 |    0.0374 |          -0.0836 |
      | 75%   |     0.6127 |   0.7476 |        0.6919 |    0.6888 |           0.6359 |
      | max   |     2.6081 |   2.5987 |        2.6845 |    2.5396 |           2.7042 |

      Standardisasi sudah berhasil dilakukan
      
# Modelling
Pada bagian ini, data yang yang sudah dibagi menjadi dua bagian menjadi _training dataset_ dan _test dataset_ siap untuk digunakan untuk pembangunan model Machine Learning-nya. Untuk kasus ini, digunakan 3 (tiga) _baseline model_ dari 3 algoritma yang berbeda. Berikut ini adalah ketiga algoritma tersebut:
- Random Forest
  - Kelebihan
    - Akurasi tinggi
    - Dapat menangani data dengan dimensi tinggi
    - Robust terhadap noise dan outliers
  - Kekurangan
    - Mahal secara komputasi
    - Butuh waktu lebih lama
    - Interpretabilitas
- KNN
  - Kelebihan
    - Sederhana dan Mudah Dipahami
    - Non-parametric
    - Tidak perlu pelatihan
  - Kekurangan
    - Sensitif terhadap outliers
    - Mahal secara komputasi
    - Memerlukan pilihan K yang baik
- SVM
  - Kelebihan
  - Kekurangan

Kemudian, baseline model dari ketiga algoritma tersebut yang memiliki akurasi tertinggi akan digunakan untuk digunakan ke tahap selanjutnya. Selanjutnya, algoritma tersebut digunakan kembali untuk pembangunan model, tetapi dengan memanfaatkan hyperparameter yang ada sehingga mendapatkan hasil terbaik. Untuk menemukan hyperparamter yang memberikan hasil terbaik, ```GridSearch``` digunakan ke model yang terpilih

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
