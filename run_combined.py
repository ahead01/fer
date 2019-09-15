import combined
import model_gen


combined_label_names = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


images, labels, freq = combined.load_combined(128, 128, 1)

model = model_gen.create_model(128, 128, 1, '', len(combined_label_names))

x_train_sp = images.reshape(len(images), 128, 128, 1).astype('float32') / 255
x_train_fq = freq.reshape(len(images), 128, 128, 1)
y_train = labels

x_test_sp = images.reshape(len(images), 128, 128, 1).astype('float32') / 255
x_test_fq = freq.reshape(len(images), 128, 128, 1)
y_test = labels

history = model.fit([x_train_sp, x_train_sp], y_train,
                    batch_size=10,
                    epochs=200,
                    validation_split=0.2)

test_scores = model.evaluate([x_test_sp, x_test_sp], y_test, verbose=0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

