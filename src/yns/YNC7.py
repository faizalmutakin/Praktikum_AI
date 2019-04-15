# In[1]:import lib
import csv #Melakukan import lib csv
from PIL import Image as pil_image #Melakukan import modul image dari lib pil
import keras.preprocessing.image #import modul image dari lib keras

# In[2]:
imgs = [] #Membuat variabel kosong imgs
classes = [] #Membuat variabel kosong Classes
with open('HASYv2/hasy-data-labels.csv') as csvfile: #Membuka file csv dari folder HASYv2
    csvreader = csv.reader(csvfile) #Membaca file csv dari variabel csvfile
    i = 0 
    for row in csvreader: # Melakukan looping dgn fungsi for in
        if i > 0:
            #Merubah file image menjadi array
            img = keras.preprocessing.image.img_to_array(pil_image.open("HASYv2/" + row[0]))
            img /= 255.0 #Mengatur skala pixel
            imgs.append((row[0], row[2], img)) #Mengisi variabel imgs
            classes.append(row[2]) #Mengisi variabel classes
        i += 1

# In[3]:
import random #Melakukan import modul random
random.shuffle(imgs) #Melakukan shuffle pada variabel imgs
split_idx = int(0.8*len(imgs)) #Melakukan split sebanyak 80% pada data train
train = imgs[:split_idx]       # Dan 20% Pada data test
test = imgs[split_idx:]

# In[4]: 

import numpy as np #Melakuan import pada modul numpy dan direname dengan np
#Melakukan input array pada data training dan testing
train_input = np.asarray(list(map(lambda row: row[2], train)))
test_input = np.asarray(list(map(lambda row: row[2], test)))
#Melakukan output array pada data training dan testing
train_output = np.asarray(list(map(lambda row: row[1], train)))
test_output = np.asarray(list(map(lambda row: row[1], test)))

# In[5]: 
from sklearn.preprocessing import LabelEncoder #Melakukan import pada modul Lavel Encoder
from sklearn.preprocessing import OneHotEncoder #Melakukan import pada modul On Hot Encoder

# In[6]:

label_encoder = LabelEncoder() #Memanggil fungsi LabelEncoder
integer_encoded = label_encoder.fit_transform(classes) #Melakukan fit pada classes
                                                       #Supaya data menjadi integer

# In[7]:
#Memanggil fungsi OneHotEncoder dimana tidak berisi matrix sparse
onehot_encoder = OneHotEncoder(sparse=False)
#Mengubah bentuk integer menjadi vektor binari dgn nilai 0 kecuali index yang diberi tanda 1
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#Melakukan fungsi fit untuk OneHotEncoder kedalam integer_encoded
onehot_encoder.fit(integer_encoded)

# In[8]:
#Mengubah data dari train_output menjadi label_encoder
train_output_int = label_encoder.transform(train_output)
#Mengubah label menjadi integer dgn OneHotEncoding dimana reshape akan memberikan bentuk array
train_output = onehot_encoder.transform(train_output_int.reshape(len(train_output_int), 1))
#Mengubah data test outpun menjadi label_encoder
test_output_int = label_encoder.transform(test_output)
#Sama seperti data training hanya saja ini menggunakan data testing
test_output = onehot_encoder.transform(test_output_int.reshape(len(test_output_int), 1))
#menampilkan jumlah data dari classes yang sudah dilakukan proses label_encoder
num_classes = len(label_encoder.classes_)
#Menampilkan kata tambahan dan akan mengembalikan nilai integer dari num_classes
print("Number of classes: %d" % num_classes)

# In[9]: import sequential
#Melakukan import model Sequential dari lib keras
from keras.models import Sequential
#Melakukan import dense, dropout, dan flatten dari modul layer yang berada pada lib keras
from keras.layers import Dense, Dropout, Flatten
#Melakukan import Conv2D dan MaxPooling2D dari modul layer yang berada pada lib keras
from keras.layers import Conv2D, MaxPooling2D
# In[10]: 
#Melakukan mpemodelan dengan model Sequential
model = Sequential()
#Melakukan konvolusi 2D dgn 32 filter yg masing-masing memiliki ukuran (3x3)
#dengan menggunakan algoritma activation relu pada data train_input dimulai dari baris 0
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=np.shape(train_input[0])))
#Menambahkan fungsi MaxPooling2D dengan pool size nya (2x2)
model.add(MaxPooling2D(pool_size=(2, 2)))
#Melakukan kovolusi 2D dgn 32 filter yg masing-masing memiliki ukuran (3x3)
#dengan menggunakan algoritma activation relu
model.add(Conv2D(32, (3, 3), activation='relu'))
#Menambahkan fungsi MaxPooling2D dengan pool size nya (2x2)
model.add(MaxPooling2D(pool_size=(2, 2)))
#Menambahkan fungsi flatten
model.add(Flatten())
#Mendefinisikan inputan dengan 1024 dense menggunakan algoritam activation tanh
model.add(Dense(1024, activation='tanh'))
#Menambahkan fungsi dropout untuk mencegah overgitting sebesar 50%
model.add(Dropout(0.5))
#Menggunakan fungsi dense untuk data num_classes dengan algotitma activation softmax
model.add(Dense(num_classes, activation='softmax'))
#Melakukan proses training dengan metode compile sebelum melatih suatu model
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
#Menampilkan representasi ringkasan model yang sudah dibuat
print(model.summary())

# In[11]: 
#Melakukan import modul callbacks dari lib keras
import keras.callbacks
#Mendefinisikan callback untuk menulis log TensorBoard yg nantinya akan memvisualisasikan grafik
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/mnist-style')

# In[12]: 5menit kali 10 epoch = 50 menit
#Melakukan pemodelan dengan data training
model.fit(train_input, train_output,
          batch_size=32, #dgn 32 ukuran subset
          epochs=10, #Melakukan ephochs sebanyak 10x
          verbose=2, #Menampilkan ephoch yang sudah berjalan
          validation_split=0.2, #Melakukan validasi split sebanyak 20%
          callbacks=[tensorboard]) #Menggunakan TensorBoards sebagai callbacks
#Menerapkan model evaluate untuk menampilkan data lost dan akurasi dari data testing
score = model.evaluate(test_input, test_output, verbose=2)
print('Test loss:', score[0]) #Menampilkan data loss pada console
print('Test accuracy:', score[1]) #Menampilkan data akurasi pada console

# In[13]:

import time #Melakukan import time dari python anaconda

results = [] #Membuat variabel results kosong
for conv2d_count in [1, 2]: #Melakukan convolution 2d yang memiliki 2 layers
    for dense_size in [128, 256, 512, 1024, 2048]: #Mendefinisikan dense size
        for dropout in [0.0, 0.25, 0.50, 0.75]: #Mendefinisikan dropout
            model = Sequential() #Menerapkan model sequential
            for i in range(conv2d_count): #Melakukan looping pada variabel conv2d_count
                #Apabila layer pertama kita perlu inputan dgn ukuran subset 32, karnel (3x3),
                # dengan menggunakan algoritma aktivation relu
                if i == 0: 
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
                #Apabila bukan kita hanya menambahkan layer saja
                else:
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
                    #Menerapkan model MaxPooling dengan ukuran (2x2)
                model.add(MaxPooling2D(pool_size=(2, 2)))
                #Menerapkan fungsi flatten
            model.add(Flatten())
            #Menambahkan dense dengan ukuran berapapun menggunakan algoritma activation tanh
            model.add(Dense(dense_size, activation='tanh'))
            #Apabila dropout digunakan kita akan menambahkan layer dropout
            if dropout > 0.0:
                model.add(Dropout(dropout)) #Menambahkan layer dropout dari variabel Dropout
            #Menambahkan dense dari num_classes dgn algoritma activation softmax
            model.add(Dense(num_classes, activation='softmax'))
            #Melakukan compile sehingga dapat kita lihat loss dan akurasinya
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #Mengatur directori log yg berbeda untuk TensorBoards sehingga dpt memberikan
            #Konvigurasi yang berbeda
            log_dir = './logs/conv2d_%d-dense_%d-dropout_%.2f' % (conv2d_count, dense_size, dropout)
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
            #Memanggil class time dari lib time
            start = time.time()
            #Menerapkan model fit atau melakukan compile terhadap data training 
            #dgn subset 32 dan ephoch 10 lalu di validation split 20%
            model.fit(train_input, train_output, batch_size=32, epochs=10,
                      verbose=0, validation_split=0.2, callbacks=[tensorboard])
            #Melakukan score dgn evaluate guna menampilkan data loss dan akurasi
            score = model.evaluate(test_input, test_output, verbose=2)
            #Menampilkan waktu akhir pada saat pemodelan terakhir
            end = time.time()
            elapsed = end - start
            #Menamilkan hasil dari source code diatasnya
            print("Conv2D count: %d, Dense size: %d, Dropout: %.2f - Loss: %.2f, Accuracy: %.2f, Time: %d sec" % (conv2d_count, dense_size, dropout, score[0], score[1], elapsed))
            results.append((conv2d_count, dense_size, dropout, score[0], score[1], elapsed))


# In[14]:
#Menerapkan model Sequential
model = Sequential()
#Menambahkan convolution dgn subset 32 dan karnel (3x3) menggunakan algoritma aktivation relu
#Pada data training
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
#Menambahkan MaxPooling dengan ukuran (2x2)
model.add(MaxPooling2D(pool_size=(2, 2)))
#Melakukan convolution pada layer kedua
model.add(Conv2D(32, (3, 3), activation='relu'))
#Menambahkan MaxPooling pada layer kedua
model.add(MaxPooling2D(pool_size=(2, 2)))
#Merapatakan inputan dengan fungsi flatten
model.add(Flatten())
#Menambahkan Dense/Neuron sebanyak 128 dgn algoritma activation tanh
model.add(Dense(128, activation='tanh'))
#Melakukan Dropout sebesar 50%
model.add(Dropout(0.5))
#Menambahkan Dense dari num_classes dgn algortima activation softmax
model.add(Dense(num_classes, activation='softmax'))
#Melakukan compile sehingga didapatkan data loss dan akurasinya
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Menampilkan Ringkasan dari model yang sudah dilakukan
print(model.summary())
# In[15]:
#melakukan fit dgn join data training dan testing agar dapat dilakukan pelatihan
model.fit(np.concatenate((train_input, test_input)),
          np.concatenate((train_output, test_output)),
          #Jumlah subsetnya adalah 32 dan ephochs 10 dan verbosenya 2
          batch_size=32, epochs=10, verbose=2)

# In[16]:
#Melakukan save dengan nama mathsymbols.model dari pemodelan yang sudah kita latih
model.save("mathsymbols.model")

# In[17]:
#Melakukan save pada label_encpder dengan na,a classes.npy
np.save('classes.npy', label_encoder.classes_)


# In[18]:
#Melakukan import moduls dari lib keras
import keras.models
#Panggil modul yang sudah kita save sebelumnya
model2 = keras.models.load_model("mathsymbols.model")
#Menampilkan ringkasan model dari mathsymbols.model
print(model2.summary())

# In[19]:
#Memanggil fungsi LabelEncoder yang sudah kita import sebelumnya
label_encoder2 = LabelEncoder()
#Meload data classes.npy
label_encoder2.classes_ = np.load('classes.npy')
#Membuat fungsi predict yang didalamnya berisikan variabel img_path
def predict(img_path):
    #Mengubah gambar dengan skala pixel 255.0 kedalam bentuk array
    newimg = keras.preprocessing.image.img_to_array(pil_image.open(img_path))
    newimg /= 255.0

    #Melakukan prediksi untuk model2 dengan bentuk array 4D
    prediction = model2.predict(newimg.reshape(1, 32, 32, 3))

    #Mencari nilai tertinggi output dari hasil prediksi
    inverted = label_encoder2.inverse_transform([np.argmax(prediction)])
    #Menampilkan hasil dari variabel prediksi dan inverted
    print("Prediction: %s, confidence: %.2f" % (inverted[0], np.max(prediction)))

# In[20]: 
#Melakukan prediksi dari pelatihan gambar v2-00010.png
predict("HASYv2/hasy-data/v2-00010.png")
#Melakukan prediksi dari pelatihan gambar v2-00500.png
predict("HASYv2/hasy-data/v2-00500.png")
#Melakukan prediksi dari pelatihan gambar v2-00700.png
predict("HASYv2/hasy-data/v2-00700.png")

# In[21]:
#Error yang saya dapat adalah seperti source code dibawah, penyebabnya menekan button error pada saat program berjalan
KeyboardInterrupt