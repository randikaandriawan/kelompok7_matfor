import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# PROGRAM K-MEANS CLUSTERING
# ANALISIS POLA BELAJAR MAHASISWA

print("\nPROGRAM K-MEANS CLUSTERING")
print("Analisis Pola Belajar Mahasiswa")

# MEMBACA DATASET

data = pd.read_csv("Dataset_Mahasiswa.csv", sep=';')

# MENAMPILKAN DATASET ASLI

print("\nDATASET MAHASISWA\n")

print(
    f"{'Nama':<15}"
    f"{'Belajar':<10}"
    f"{'Coding':<10}"
    f"{'Menunda':<10}"
    f"{'MediaSosial':<15}"
    f"{'Diskusi':<10}"
    f"{'AI'}"
)

print("-" * 80)

for index, row in data.iterrows():

    print(
        f"{row['Nama']:<15}"
        f"{row['Belajar']:<10}"
        f"{row['Coding']:<10}"
        f"{row['Menunda']:<10}"
        f"{row['MediaSosial']:<15}"
        f"{row['Diskusi']:<10}"
        f"{row['AI']}"
    )

print("\nKETERANGAN:")
print("- Atribut Menunda dan MediaSosial merupakan atribut negatif")
print("- Semakin tinggi nilainya maka semakin kurang baik")
print("- Sebelum proses clustering dilakukan, kedua atribut tersebut dibalik menggunakan rumus:")
print("  nilai baru = 6 - nilai lama")
print("- Tujuannya agar seluruh atribut memiliki arah penilaian yang sama")
print("- Setelah dibalik, semakin besar nilai maka semakin baik")


# PREPROCESSING DATA
# Menunda dan Media Sosial dibalik
# karena termasuk atribut negatif

data_proses = data.copy()

data_proses['Menunda'] = 6 - data_proses['Menunda']
data_proses['MediaSosial'] = 6 - data_proses['MediaSosial']

# MEMILIH ATRIBUT

X = data_proses[[
    'Belajar',
    'Coding',
    'Menunda',
    'MediaSosial',
    'Diskusi',
    'AI'
]]

# MENENTUKAN CENTROID AWAL

initial_centroids = np.array([
    [5, 5, 4, 3, 5, 5],  # Cluster 1 = aktif
    [3, 2, 2, 1, 4, 4],  # Cluster 2 = sedang
    [1, 1, 1, 1, 3, 3]   # Cluster 3 = kurang konsisten
])

# MENAMPILKAN CENTROID AWAL

print("\nCENTROID AWAL\n")

print(
    f"{'Cluster':<12}"
    f"{'Belajar':<10}"
    f"{'Coding':<10}"
    f"{'Menunda':<10}"
    f"{'MediaSosial':<15}"
    f"{'Diskusi':<10}"
    f"{'AI'}"
)

print("-" * 80)

for i, centroid in enumerate(initial_centroids, start=1):

    print(
        f"{'Cluster '+str(i):<12}"
        f"{centroid[0]:<10}"
        f"{centroid[1]:<10}"
        f"{centroid[2]:<10}"
        f"{centroid[3]:<15}"
        f"{centroid[4]:<10}"
        f"{centroid[5]}"
    )

# IMPLEMENTASI K-MEANS

kmeans = KMeans(
    n_clusters=3,
    init=initial_centroids,
    n_init=1,
    random_state=42
)

kmeans.fit(X)


# MENAMBAHKAN HASIL CLUSTER

data['Cluster'] = kmeans.labels_ + 1

# MENYESUAIKAN NOMOR CLUSTER

mapping_cluster = {
    1: 1,  # aktif tetap 1
    2: 3,  # sklearn 2 jadi cluster 3
    3: 2   # sklearn 3 jadi cluster 2
}

data['Cluster'] = data['Cluster'].map(mapping_cluster)

# HASIL CLUSTERING

print("\nHASIL PENGELOMPOKAN DATA\n")

print(
    f"{'Nama':<15}"
    f"{'Belajar':<10}"
    f"{'Coding':<10}"
    f"{'Menunda':<10}"
    f"{'MediaSosial':<15}"
    f"{'Diskusi':<10}"
    f"{'AI':<5}"
    f"{'Cluster'}"
)

print("-" * 90)

for index, row in data.iterrows():

    print(
        f"{row['Nama']:<15}"
        f"{row['Belajar']:<10}"
        f"{row['Coding']:<10}"
        f"{row['Menunda']:<10}"
        f"{row['MediaSosial']:<15}"
        f"{row['Diskusi']:<10}"
        f"{row['AI']:<5}"
        f"{row['Cluster']}"
    )

# ANGGOTA SETIAP CLUSTER

print("\nANGGOTA SETIAP CLUSTER")
for i in range(1, 4):

    print(f"\nCluster {i}")

    if i == 1:
        print("Karakteristik : Mahasiswa aktif belajar")
    elif i == 2:
        print("Karakteristik : Mahasiswa dengan pola belajar sedang")
    elif i == 3:
        print("Karakteristik : Mahasiswa kurang konsisten dalam belajar")
    print()

    cluster_data = data[data['Cluster'] == i]

    for nama in cluster_data['Nama']:

        print(f"- {nama}")


# JUMLAH ANGGOTA CLUSTER


print("\nJUMLAH ANGGOTA CLUSTER\n")

jumlah = data['Cluster'].value_counts().sort_index()

for cluster, total in jumlah.items():

    print(f"Cluster {cluster} : {total} mahasiswa")


# CENTROID AKHIR


centroid = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=[
        'Belajar',
        'Coding',
        'Menunda',
        'MediaSosial',
        'Diskusi',
        'AI'
    ]
)

centroid.index = [
    'Cluster 1',
    'Cluster 2',
    'Cluster 3'
]

print("\nCENTROID AKHIR\n")

print(
    f"{'Cluster':<12}"
    f"{'Belajar':<10}"
    f"{'Coding':<10}"
    f"{'Menunda':<10}"
    f"{'MediaSosial':<15}"
    f"{'Diskusi':<10}"
    f"{'AI'}"
)

print("-" * 80)

for index, row in centroid.iterrows():

    print(
        f"{index:<12}"
        f"{row['Belajar']:<10.2f}"
        f"{row['Coding']:<10.2f}"
        f"{row['Menunda']:<10.2f}"
        f"{row['MediaSosial']:<15.2f}"
        f"{row['Diskusi']:<10.2f}"
        f"{row['AI']:.2f}"
    )

