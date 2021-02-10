import tensorflow as tf
import imutils
from imutils.object_detection import non_max_suppression
import cv2
import base64
import time
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class_idx = ['0','1','2','3','4','5','6','7','8','9',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
             'a','b','d','e','f','g','h','n','q','r','t']
# class_idx = ['0','1','2','3','4','5','6','7','8','9',
#              'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
#              'a','b','c','d','e','f','g','h','i','j','k','l','n','m', 'o', 'p', 'q','r', 's','t', 'u', 'v', 'w', 'x', 'y', 'z']

def sliding_window(image, step, ws):
    """ slide a window across the image """
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid(image, scale=1.5, minSize=(28, 28)):
    """generate the image pyramid"""
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def run_inference(base64_image):
    WIN_STEP = 8
    ROI_SIZE = (100, 150)
    INPUT_SIZE = (28, 28)
    # decode the base64 image
    encoded_data = base64_image.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = imutils.resize(image, width=600)
    image = cv2.bitwise_not(image)
    # change to gray colour
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = image.shape[:2]
    # initialize the image pyramid
    pyramid = image_pyramid(image)
    rois = []
    locs = []
    start = time.time()
    model = load_model('aaaaav3.h5')
    # loop over the image pyramid
    scale = W / float(image.shape[1])

    # for each layer of the image pyramid, loop over the sliding
    # window locations
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # scale the (x, y)-coordinates of the ROI with respect to the
        # *original* image dimensions
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        # take the ROI and preprocess it so we can later classify
        # the region using Keras/TensorFlow
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)

        # update our list of ROIs and associated coordinates
        rois.append(roi)
        locs.append((x, y, x + w, y + h))
        # check to see if we are visualizing each of the sliding
        # windows in the image pyramid
        # clone the original image and then draw a bounding box
        # surrounding the current region
        clone = image.copy()
        # cv2.rectangle(clone, (x, y), (x + w, y + h),
        #               (0, 255, 0), 2)
        #
        # # show the visualization and current ROI
        # cv2.imshow("Visualization", clone)
        # cv2.imshow("ROI", roiOrig)
        # cv2.waitKey(0)
    end = time.time()
    print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
        end - start))

    # convert the ROIs to a NumPy array
    rois = np.array(rois, dtype="float32")

    # classify each of the proposal ROIs using ResNet and then show how
    # long the classifications took
    print("[INFO] classifying ROIs...")
    start = time.time()

    preds = model.predict(rois)
    end = time.time()
    print("[INFO] classifying ROIs took {:.5f} seconds".format(
        end - start))

    # decode the predictions and initialize a dictionary which maps class
    # labels (keys) to any ROIs associated with that label (values)
    readable_predictions = []
    labels = {}
    # loop through the predicted values
    for count, pred in enumerate(preds):
        # get the max result index and the confidence value from that
        max_result_idx = np.argmax(pred)
        max_confidence = pred[max_result_idx]
        # filter out the lousy predictions
        if max_confidence > 0.9:
            # add it to the list of readable predictions
            L = labels.get(class_idx[max_result_idx], [])
            L.append((locs[count], max_confidence))
            labels[class_idx[max_result_idx]] = L
            readable_predictions.append((locs[count], class_idx[max_result_idx]))
    clone = image.copy()
    # print(readable_predictions)
    for label in labels.keys():
        # clone = image.copy()
        clone = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)

        # loop over all bounding boxes that were kept after applying
        # non-maxima suppression
        for (startX, startY, endX, endY) in boxes:
            # draw the bounding box and label on the image
            cv2.rectangle(clone, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # show the output after apply non-maxima suppression
        cv2.imshow("After", clone)
        cv2.waitKey(0)
    # loop through the readable predictions to label them and perform non_max_suppression
    for ((idx, (locs, img_class))) in enumerate(readable_predictions):
        boxes = np.array([p[0] for p in readable_predictions[idx]])
        proba = np.array([p[1] for p in readable_predictions[idx]])
        boxes = non_max_suppression(boxes, proba)
        (startX, startY, endX, endY) = locs
        cv2.rectangle(clone, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, img_class, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("Before", clone)
    cv2.waitKey(0)


def run_inference_opencv(base64_image):
    encoded_data = base64_image.split(',')[1]
    # load the machine learning model
    model = load_model('c_model_43620.h5')
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = imutils.resize(image, width=600)
    # change to gray colour
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # find the contours (continuous blobs of pixels) the image
    contours = find_contours(image_gray)
    # find the number of usable found contours using list comprehension (fast)
    contours = [contour for contour in contours if cv2.contourArea(contour) <= 7000]
    # invert image colour if we did not find 3 or more contours (captchas are typically 4 digits or more)
    # this happens as contour finding only looks for contours that are white on a black bg
    if len(contours) == 0:
        image_gray = cv2.bitwise_not(image_gray)
        contours = find_contours(image_gray)

    # initialize an empty array to store letter positions
    letter_positions = []
    for contour in contours:
        # get the rectangle in the contour
        print(f"contour size: {cv2.contourArea(contour)}")
        (x, y, w, h) = cv2.boundingRect(contour)
        # append the position as a tuple to the list of positions
        x = x-7
        if x < 0:
            x = 0
        letter_positions.append((x, y-7, w+7, h+7))

    # sort from left to right
    letter_positions = sorted(letter_positions, key=lambda x: x[0])
    letters = []
    for pos in letter_positions:
        (x, y, w, h) = pos
        # MNIST dataset is white on black bg, invert our image.
        image_gray = cv2.bitwise_not(image_gray)
        letter_img = image_gray[y:y + h, x:x + w]
        letter_img = cv2.copyMakeBorder(letter_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0,0,0])
        letter_img = cv2.resize(letter_img, (28, 28))
        letter_img = img_to_array(letter_img)
        letters.append(letter_img)
    start_time = time.time()
    letters = np.array(letters, dtype="float16")
    preds = model.predict(letters)
    readable_predictions = ""
    for count, prediction in enumerate(preds):
        (x, y, w, h) = letter_positions[count]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        pred_idx = np.argmax(prediction)
        cv2.putText(image, class_idx[pred_idx], (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        readable_predictions += class_idx[pred_idx]
    print(f"prediction took: {time.time() - start_time}s")
    print(readable_predictions)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return readable_predictions


def find_contours(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Hack for compatibility with different OpenCV versions
    return contours[1] if imutils.is_cv3() else contours[0]


if __name__ == '__main__':
    run_inference_opencv(" data:image/png;base64,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")