import cv2
import tensorflow as tf
import pre_processing

def top_eyes(eyes):
    top_eyes = []
    for eye in eyes:
        if len(top_eyes) == 0:
            top_eyes.append(eye)
            continue
        if len(top_eyes) == 1:
            top_eyes.append(eye)
            if eye[0] < top_eyes[0][0]:
                top_eyes[1] = top_eyes[0]
                top_eyes[0] = eye
            continue
        d1 = abs(eye[1] - top_eyes[0][1])
        d2 = abs(eye[1] - top_eyes[1][1])
        d3 = abs(top_eyes[0][1] - top_eyes[1][1])
        min_dist = min(d1, d2, d3)
        if d1 == min_dist:
            if top_eyes[0][0] > eye[0]:
                top_eyes[1] = top_eyes[0]
                top_eyes[0] = eye
            else:
                top_eyes[1] = eye
        elif d2 == min_dist:
            if top_eyes[1][0] < eye[0]:
                top_eyes[0] = top_eyes[1]
                top_eyes[1] = eye
            else:
                top_eyes[0] = eye
        elif d3 == min_dist:
            continue
    return top_eyes

def get_face(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        face_img = img[y:y+h, x:x+w]
        return(face_img)

def get_eyes(img):
    eyes = eye_cascade.detectMultiScale(img)
    # If more than 2 eyes find the two closest on the y plane
    if len(eyes) > 2:
        #eyes = top_eyes(eyes)
        indxs = []
        for eye in eyes:
            indxs.append(eye[1])
        print(type(indxs))
        print(indxs)
        val, idx1 = min((val, idx) for (idx, val) in enumerate(indxs))
        print(val, idx1)
        indxs[idx1] = max(indxs)
        val, idx2 = min((val, idx) for (idx, val) in enumerate(indxs))
        print(val, idx2)
        tmp = [eyes[idx1], eyes[idx2]]
        if tmp[0][0] > tmp[1][0]:
            tmp2 = tmp
            tmp = [tmp2[1], tmp2[0]]
        eyes = tmp
    if len(eyes) == 2:
        if eyes[0][0] > eyes[1][0]:
                tmp2 = eyes
                eyes = [tmp2[1], tmp2[0]]

    count = 1
    eye_imgs = []
    for (ex,ey,ew,eh) in eyes:
        eye_img = img[ey:ey+eh, ex:ex+ew]
        eye_imgs.append(eye_img)
        #print(eye_img.shape)
        cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (count * 255,255,0), 2)
        count -= 1
    return eye_imgs[0], eye_imgs[1], img

if __name__ == '__main__':
    DATA_DIR = "C:\\Users\\Austin\\Pictures\\jaffe\jaffe\\"

    PROJECT_DIR = 'C:\\Users\\Austin\\Desktop\\gPrj\\fer'
    cascade_dir = PROJECT_DIR + '/FaceDetection/xml'
    face_cascade_xml = cascade_dir + '/haarcascade_frontalface_default.xml'
    eye_cascade_xml = cascade_dir + '/haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(face_cascade_xml)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_xml)


    image_name = 'YM.SU3.60.tiff'
    image_path = DATA_DIR + image_name

    image = cv2.imread(image_path)

    face_image = get_face(image)

    pre_processing.show_image(face_image, 'Face')

    labeled_face, left_eye, right_eye = get_eyes(face_image)

    pre_processing.show_image(labeled_face, "Labeled Face")
    pre_processing.show_image(left_eye, 'Left Eye')
    pre_processing.show_image(right_eye, 'Right Eye')
    
    