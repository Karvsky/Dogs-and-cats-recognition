import os
import sys
import cv2
from contextlib import contextmanager

@contextmanager
def suppress_stderr():

    original_stderr_fd = sys.stderr.fileno()

    saved_stderr_fd = os.dup(original_stderr_fd)

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        os.dup2(devnull_fd, original_stderr_fd)

        os.close(devnull_fd)
        yield
    finally:

        os.dup2(saved_stderr_fd, original_stderr_fd)

        os.close(saved_stderr_fd)
def data_import():
    directory = r'Dogs and cat recognition\Dataset\PetImages'
    categories = ["Dog", "Cat"]
    img_array = []
    img_labels = []

    for i in categories:
        path = os.path.join(directory, i)
        for img in os.listdir(path):
            image_path = os.path.join(path, img)

            with suppress_stderr():
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                new_image = cv2.resize(image, (70, 50))
                img_array.append(new_image)
                if i == 'Cat':
                    img_labels.append(0)
                else:
                    img_labels.append(1)
            else:
                pass
    return img_array, img_labels

