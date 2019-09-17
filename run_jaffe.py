import jaffe
import model_gen
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

HEIGHT, WIDTH, CHANNELS = 128, 128, 1

label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

images, labels, freq = jaffe.for_jaffe(HEIGHT, WIDTH, CHANNELS)

model = model_gen.create_deep_model(HEIGHT, WIDTH, CHANNELS, 20, len(label_names))

x_train_sp = images.reshape(len(images), HEIGHT, WIDTH, CHANNELS).astype('float32') / 255
x_train_fq = freq.reshape(len(images), HEIGHT, WIDTH, CHANNELS)
y_train = labels

X_train = x_train_sp
X_test = x_train_sp
y_test = y_train
#X_train, X_test, y_train, y_test = train_test_split(x_train_sp, y_train, train_size=0.75, shuffle=True)

history = model.fit([X_train, x_train_fq], y_train,
                    batch_size=10,
                    epochs=70
                    ,validation_split=0.2)

test_scores = model.evaluate([X_test, x_train_fq], y_test, verbose=0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
