import combined
import model_gen
import matplotlib.pyplot as plt

HEIGHT, WIDTH, CHANNELS = 128, 128, 1

combined_label_names = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


images, labels, freq = combined.load_combined(128, 128, 1)

model = model_gen.create_deep_model(HEIGHT, WIDTH, CHANNELS, 20, len(combined_label_names))

x_train_sp = images.reshape(len(images), HEIGHT, WIDTH, CHANNELS).astype('float32') / 255
x_train_fq = freq.reshape(len(images), HEIGHT, WIDTH, CHANNELS)
y_train = labels

x_test_sp = images.reshape(len(images), HEIGHT, WIDTH, CHANNELS).astype('float32') / 255
x_test_fq = freq.reshape(len(images), HEIGHT, WIDTH, CHANNELS)
y_test = labels

history = model.fit([x_train_sp, x_train_sp], y_train,
                    batch_size=10,
                    epochs=200,
                    validation_split=0.2)

test_scores = model.evaluate([x_test_sp, x_test_sp], y_test, verbose=0)
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
