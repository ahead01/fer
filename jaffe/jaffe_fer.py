import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pre_processing
import jaffe
import model_gen

# Globals
label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']

CASCADE_DIR = PROJECT_DIR + '/FaceDetection/xml'
FACE_XML = CASCADE_DIR + '/haarcascade_frontalface_default.xml'
EYE_XML = CASCADE_DIR + '/haarcascade_eye.xml'


FACE_CASCADE = cv2.CascadeClassifier(FACE_XML)
EYE_CASCADE = cv2.CascadeClassifier(EYE_XML)

def select_eyes(eyes):
    if len(eyes) == 0:
        return None
    if len(eyes) > 2:
        #eyes = top_eyes(eyes)
        indxs = []
        for eye in eyes:
            indxs.append(eye[1])

        val, idx1 = min((val, idx) for (idx, val) in enumerate(indxs))
        indxs[idx1] = max(indxs)
        val, idx2 = min((val, idx) for (idx, val) in enumerate(indxs))

        tmp = [eyes[idx1], eyes[idx2]]
        if tmp[0][0] > tmp[1][0]:
            tmp2 = tmp
            tmp = [tmp2[1], tmp2[0]]
        eyes = tmp

    if len(eyes) == 2:
        if eyes[0][0] > eyes[1][0]:
                tmp2 = eyes
                eyes = [tmp2[1], tmp2[0]]
        return eyes

    if len(eyes) < 2:
        return None

def load_hap_images(img_height, img_width, img_chan, pattern, edge_images=False):
    ''' height, width, channels'''
    img_dir = DATA_DIR + '/jaffe/'
    tiff_pattern = re.compile('\.tiff', re.IGNORECASE)
    pattern = hap_ptr = re.compile(pattern)
    # Count files in dir so I can pre allocate np arrays
    image_count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            image_count += 1
    # Allocate arrays
    labels = np.empty((208), dtype=np.uint8)
    images = np.empty((208, img_height, img_width), dtype=np.uint8)
    eye_imgs = np.zeros((208, 128, 64), dtype=np.uint8)
    count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            # Load image unchanged
            img = cv2.imread(img_dir + file_name, cv2.IMREAD_UNCHANGED)
            faces = FACE_CASCADE.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in faces:
                face_img = img[y:y+h, x:x+w]
            img = cv2.resize(face_img, (img_height, img_width))
            eyes = EYE_CASCADE.detectMultiScale(face_img)
            eyes = select_eyes(eyes)
            if eyes is None:
                continue
            tmp_eyes = []
            for (ex,ey,ew,eh) in eyes:
                eye_img = face_img[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (64, 64))
                tmp_eyes.append(eye_img)
            eye_img = np.concatenate((tmp_eyes[0], tmp_eyes[1]), axis=0)

            eye_imgs[count] = eye_img
            # Detect edges
            if edge_images:
                img = pre_processing.edge_detection(img)
            images[count] = img
            if pattern.search(file_name):
                labels[count] = 1
            else:
                labels[count] = 0
            count += 1

    print(count)

    return images, labels, eye_imgs

def load_images(img_height, img_width, img_chan, edge_images=False):
    ''' height, width, channels'''
    img_dir = DATA_DIR + '/jaffe/'
    tiff_pattern = re.compile('\.tiff', re.IGNORECASE)
    patterns = jaffe.get_patterns()

    # Count files in dir so I can pre allocate np arrays
    image_count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            image_count += 1

    # Allocate arrays
    labels = np.empty((208), dtype=np.uint8)
    images = np.empty((208, img_height, img_width), dtype=np.uint8)
    eye_imgs = np.zeros((208, 128, 64), dtype=np.uint8)
    count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            # Load image unchanged
            img = cv2.imread(img_dir + file_name, cv2.IMREAD_UNCHANGED)
            faces = FACE_CASCADE.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in faces:
                face_img = img[y:y+h, x:x+w]
            img = cv2.resize(face_img, (img_height, img_width))
            eyes = EYE_CASCADE.detectMultiScale(face_img)
            eyes = select_eyes(eyes)
            if eyes is None:
                continue
            tmp_eyes = []
            for (ex,ey,ew,eh) in eyes:
                eye_img = face_img[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (64, 64))
                tmp_eyes.append(eye_img)
            eye_img = np.concatenate((tmp_eyes[0], tmp_eyes[1]), axis=0)

            eye_imgs[count] = eye_img
            # Detect edges
            if edge_images:
                img = pre_processing.edge_detection(img)
            images[count] = img
            for label, pattern in enumerate(patterns):
                if pattern.search(file_name):
                    labels[count] = label
                    count += 1
                    break
    print(count)

    return images, labels, eye_imgs







if __name__ == '__main__':
    img_height = 256
    img_width = 256
    eye_height = 128
    eye_width = 64

    if True:
        images, labels, eye_imgs = load_images(img_height, img_width, 1, edge_images=True)

        x_train, x_test, y_train, y_test = train_test_split(images, labels,  shuffle=True)

        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        print(x_test.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        #pre_processing.show_image(images[22], label_names[int(labels[22])])

        #pre_processing.show_image(eye_imgs[22], label_names[int(labels[22])])

        n_classes = len(np.unique(labels))
        print(f'There are {n_classes} classes.')

    if False:
        model = model_gen.create_simple_model(eye_height, eye_width, '20', n_classes)
        history = model.fit(eye_imgs, labels,
                        batch_size=10,
                        epochs=70
                        ,validation_split=0.2)

        test_scores = model.evaluate(eye_imgs, labels, verbose=0)
        print('Test loss:', test_scores[0])
        print('Test accuracy:', test_scores[1])

    if False:
        model = model_gen.create_simple_model(img_height, img_width, '20', n_classes)
        history = model.fit(images, labels,
                        batch_size=10,
                        epochs=50
                        ,validation_split=0.2)

        test_scores = model.evaluate(images, labels, verbose=0)
        print('Test loss:', test_scores[0])
        print('Test accuracy:', test_scores[1])

    if False:
        model = model_gen.face_and_eye_model(img_height, img_width, eye_height, eye_width, n_classes)

        pre_processing.show_image(eye_imgs[35], label_names[int(labels[35])])

        history = model.fit([images, eye_imgs], labels,
                        batch_size=10,
                        epochs=35
                        ,validation_split=0.2)

        test_scores = model.evaluate([images, eye_imgs], labels, verbose=0)
        print('Test loss:', test_scores[0])
        print('Test accuracy:', test_scores[1])

    #Only happy images model
    if False:
        images, labels, eye_imgs = load_hap_images(img_height, img_width, 1, 'NE', edge_images=True)

        n_classes = len(np.unique(labels))
        print(f'There are {n_classes} classes.')

        positives = images[labels == 1]
        negatives = images[labels == 0]
        p_labels = labels[labels == 1]
        n_labels = labels[labels == 0]

        data = np.concatenate((positives[:30], negatives[:30]), axis=0)
        labels = np.concatenate((p_labels[:30], n_labels[:30]), axis=0)

        x_train, x_test, y_train, y_test = train_test_split(data, labels,  shuffle=True)

        if True:
            model = model_gen.hap_model(img_height, img_width, n_classes)

            model = model_gen.compile_model(model)

            history = model.fit(x_train, y_train,
                            batch_size=1,
                            epochs=25
                            ,validation_split=0)

            test_scores = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', test_scores[0])
            print('Test accuracy:', test_scores[1])

            model_gen.save_trained_model(model, 'jaffe_neu_002')
            print(y_train, '\\', y_test)

    if False:
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
