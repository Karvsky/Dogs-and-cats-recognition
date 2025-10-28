from data_import import data_import


def normalization():

    img_array, img_labels = data_import()

    normalized_dataset = []

    for i in img_array:
        image_matrix_float = i.astype('float32')
        normalized_dataset.append(image_matrix_float / 255.0)

    return normalized_dataset
