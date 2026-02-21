import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tensorflow as tf
import sklearn
from keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization, Dropout, MaxPooling2D, Concatenate
from keras import Model
import pickle
import collections
import seaborn as sns

############## Data importation ##############
ds_train, info = tfds.load('speech_commands', split='train[0:80%]', as_supervised=True, with_info=True)
ds_val_raw = tfds.load('speech_commands', split='train[80%:]', as_supervised=True)

label_name = info.features['label'].names

############## Data preparation ##############
# Class frequency 
def frequency_count(ds):    
    counter = collections.Counter()
    for _, label in ds:
        counter[label_name[label.numpy()]] += 1
    
    for i in counter:
        print(f"{i} : {counter[i]/counter.total()*100:.2f}%")

    return counter

#counter = frequency_count(ds_train)

unknown_id = label_name.index('_unknown_')

def add_noise(audio, label):
    audio = tf.cast(audio, tf.float32)
    noise = tf.random.normal(shape=tf.shape(audio), mean=0.0, stddev=0.005)
    audio = audio + noise 
    return audio, label

mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins = 64,
    num_spectrogram_bins = 257,
    sample_rate = 16000,
    lower_edge_hertz=20.0,
    upper_edge_hertz=8000.0
    )

def preprocess(audio, label):
    # Padding to size 16,000
    audio = tf.cast(audio, tf.float32)
    audio = audio[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], axis=0)
    
    # Normalization and calculation of Fourrier transform
    audio_proc = audio / tf.reduce_max(tf.abs(audio))
    stft = tf.signal.stft(audio_proc, frame_length=320, frame_step=160)
    stft_spectrogram = tf.abs(stft)
      
    # Calculating Mel spectrogram
    mel_spectrogram = tf.matmul(stft_spectrogram, mel_matrix)

    spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
       
    return spectrogram, label

def filter_maxLabel(audio, label):
    # Used to lower the percentage of data belonging to the class Unknown
    is_unknown = tf.equal(label, unknown_id)
    keep_random = tf.random.uniform([], 0, 1) < 0.3 # keep 30% of the initial unknown class
    return tf.logical_or(tf.logical_not(is_unknown), keep_random)


ds_train_balanced = ds_train.filter(filter_maxLabel) \
    .map(add_noise,num_parallel_calls=tf.data.AUTOTUNE) \
    .map(preprocess,num_parallel_calls=tf.data.AUTOTUNE) \
    .shuffle(buffer_size=1000) \
    .cache() \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE)
    
ds_val = ds_val_raw.map(lambda audio, label: (tf.cast(audio, tf.float32), label),num_parallel_calls=tf.data.AUTOTUNE) \
    .map(preprocess,num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE)
    
class AudioCNNModel:
   def __init__(self, num_classes, input_dim=16000):
       self.num_classes = num_classes
       self.input_dim = input_dim
       
       self.training_accuracy = []
       self.training_loss = []
       self.validation_accuracy = []
       self.validation_loss = []


   def model(self): 
       inputs = Input(shape=(99, 64, 1), name='Inputs')

       x_time = Conv2D(filters=16, kernel_size=(1,5), strides = 1, padding='same', activation='relu')(inputs)
       x_frequency = Conv2D(filters=16, kernel_size=(5,1), strides = 1, padding='same', activation='relu')(inputs)
       x = Concatenate()([x_time, x_frequency])
       x = BatchNormalization()(x)
       x = MaxPooling2D(pool_size=(2, 2))(x)
       
       x = Conv2D(filters=32, kernel_size=3, strides = 1, padding='same', activation='relu')(x)
       x = BatchNormalization()(x)
       x = MaxPooling2D(pool_size=(2, 2))(x)
         
       x = Conv2D(filters=64, kernel_size=3, strides = 1, padding='same', activation='relu')(x)
       x = BatchNormalization()(x)
       x = MaxPooling2D(pool_size=(2, 2))(x)
         
       x = Conv2D(filters=128, kernel_size=3, strides = 1, padding='same', activation='relu')(x)
       x = BatchNormalization()(x)
       x = MaxPooling2D(pool_size=(2, 2))(x)
         
       x = Flatten()(x)
       x = Dense(128, activation='relu')(x)
         
       x = Dropout(0.3)(x)
       outputs = Dense(self.num_classes, activation='softmax')(x)
         
       self.model = Model(inputs, outputs, name="AudioCNN")
       
   def f_compile(self, learning_rate):

       optimizer=tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
       self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

   def f_train(self, ds_train, ds_test):
       from keras.callbacks import EarlyStopping
       
       history = self.model.fit(
           x=ds_train,
           validation_data = ds_test,
            shuffle = True,
            epochs=30,
            callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
            )
       
       self.training_accuracy.extend(history.history['accuracy'])
       self.training_loss.extend(history.history['loss'])
       self.validation_accuracy.extend(history.history['val_accuracy'])
       self.validation_loss.extend(history.history['val_loss'])

############ Model ############
Audio_Model = AudioCNNModel(num_classes = len(label_name))
print("\nParameters initialized")
Audio_Model.model()
print("\nModel created")
Audio_Model.f_compile(learning_rate = 0.001)
print("\nModel compile")

loading = 1
if loading:
    file = 'Model_Audio'
    with open(file +'.pkl', 'rb') as f:
        loaded_attributes = pickle.load(f)

    Audio_Model.training_accuracy = loaded_attributes['training_accuracy']
    Audio_Model.training_loss = loaded_attributes['training_loss']
    Audio_Model.validation_accuracy = loaded_attributes['validation_accuracy']
    Audio_Model.validation_loss = loaded_attributes['validation_loss']

    Audio_Model.model.load_weights(file + '.keras')
    print("\nLoaded")

print("\nTraining ... ")
# Audio_Model.f_train(ds_train_balanced, ds_val)

############ Results ############
results = Audio_Model.model.evaluate(ds_val)

print(f"Loss : {results[0]}")
print(f"Accuracy : {results[1] * 100:.2f}%")

## Plotting losses and accuracy
plt.figure()
plt.title("Accuracy")
plt.xlabel('epoch')
plt.plot(Audio_Model.training_accuracy, '--', color='blue')
plt.plot(Audio_Model.validation_accuracy, '-', color='blue')
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.ylim(0, 1)

plt.figure()
plt.title("Losses")
plt.xlabel('epoch')
plt.plot(Audio_Model.training_loss,'--', color='orange')
plt.plot(Audio_Model.validation_loss, '-', color='orange')
plt.legend(['Training loss', 'Validation loss'])
plt.ylim(0, 1.5)


############ Confusion Matrix ############
y_pred = []
y_true = []
for audio, label in ds_val:
    y_true.extend(label.numpy())
    
    prediction = Audio_Model.model.predict(audio, verbose = 0)
    
    y_pred.extend(np.argmax(prediction, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmax = 1000)

plt.xlabel('Predicated Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Audio Recognition')
plt.show()

## Score
Report = sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
print(Report)

plt.figure(figsize=(6, 4))
plt.suptitle("Mel Spectrogram")
i = 0
for audio, label in ds_val_raw.take(25):
    audio_numpy = audio.numpy()
        
    audio_padded = np.pad(audio_numpy[:16000], (0,16000-len(audio_numpy)), mode='constant', constant_values=0)

    volume=0.5
    sd.play(audio_padded*volume/np.max(np.abs(audio_padded)), 16000)
    sd.wait()

    spectrogram = preprocess(audio_padded, label)[0]
    input_tensor = np.expand_dims(spectrogram, axis=0)
    prediction = Audio_Model.model.predict(input_tensor, verbose = 0)
    indice = np.argmax(prediction)

    print(f"True Label ID : {label} ({label_name[label]})")
    print(f"Label Prediction : {label_name[indice]}")
    
    ax = plt.subplot(5, 5, i + 1)        
    ax.imshow(input_tensor.squeeze().T, aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    ax.set_title(label_name[label], fontsize=10)
    i+=1
plt.tight_layout()

plt.show()            

## Saving
save=0
if save:
    fil_name = "Model_Audio"
    Audio_Model.model.save(fil_name+'.keras')

    attributes = {
        'training_accuracy': Audio_Model.training_accuracy,
        'training_loss': Audio_Model.training_loss,
        'validation_accuracy': Audio_Model.validation_accuracy,
        'validation_loss': Audio_Model.validation_loss,
    }
    with open(fil_name+'.pkl', 'wb') as f:
        pickle.dump(attributes, f)
        
    print("Model saved")
