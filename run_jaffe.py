import jaffe
import model_gen
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

HEIGHT, WIDTH, CHANNELS = 128, 128, 1

label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

images, labels, freq = jaffe.for_jaffe(HEIGHT, WIDTH, CHANNELS)

model = model_gen.create_conv_model(HEIGHT, WIDTH, CHANNELS, '', len(label_names))

x_train_sp = images.reshape(len(images), HEIGHT, WIDTH, CHANNELS).astype('float32') / 255
x_train_fq = freq.reshape(len(images), HEIGHT, WIDTH, CHANNELS)
y_train = labels

X_train, X_test, y_train, y_test = train_test_split(x_train_sp, y_train, train_size=0.75, shuffle=True)

history = model.fit([X_train], y_train,
                    batch_size=10,
                    epochs=200)
                    #,validation_split=0.2)

test_scores = model.evaluate([X_test], y_test, verbose=0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
