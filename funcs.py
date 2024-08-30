import cv2 as cv
import seaborn as sns
import numpy as np

# from google.colab.patches import cv2_imshow


from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
from math import sqrt


def cv_imread(path):
    image = cv.imread(path)
    if image is None:
        raise ValueError(f'File "{path}" does not exist')
    return image


def cv_grayread(path, asfloat=False):
    image = cv_imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if asfloat:
        image = image.astype(np.float32)
    return image


def cv_imshow(image):    
    
    
    # Convert from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.show()
    # return cv2_imshow(image)


def cv_gridshow(images, start=0, stop=9, labels=None, cmap=None, lmap=None):
    images = images[start:stop]
    num_imgs = len(images)
    if num_imgs == 0:
        raise ValueError('Number of images cannot be zero')

    if labels is not None:
        labels = labels[start:stop]
        if len(labels) != num_imgs:
            raise ValueError('Number of images and labels must be equal')

    num_cols = int(sqrt(num_imgs))
    num_rows = num_imgs // num_cols
    if num_imgs % num_cols != 0:
        num_rows += 1

    indices = range(1, num_imgs + 1)

    plt.figure(figsize=(num_rows, num_cols))

    if labels is None:
        for image, index in zip(images, indices):
            if cmap is None:
                if len(image.shape) == 2:
                    image_cmap = 'gray'
                else:
                    image = image[:, :, ::-1]
                    image_cmap = 'viridis'
            else:
                image_cmap = cmap
            plt.subplot(num_rows, num_cols, index)
            plt.imshow(image, image_cmap)
            plt.axis("off")
    else:
        for image, label, index in zip(images, labels, indices):
            if cmap is None:
                if len(image.shape) == 2:
                    image_cmap = 'gray'
                else:
                    image = image[:, :, ::-1]
                    image_cmap = 'viridis'
            else:
                image_cmap = cmap
            if lmap is None:
                def lmap(label): return label
            plt.subplot(num_rows, num_cols, index)
            plt.imshow(image, image_cmap)
            plt.title(lmap(label))
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion(model, test_images, test_labels, scale=1, lmap=None):
    if lmap is None:
        def lmap(label): return label
    labels = [lmap(np.argmax(p)) for p in model.predict(test_images)]
    matrix = confusion_matrix(test_labels, labels)
    rcParams['figure.figsize'] = (scale * 6, scale * 5)
    sns.heatmap(matrix, annot=True, fmt='g')


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'])


sns.set()
