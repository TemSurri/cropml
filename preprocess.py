import tensorflow as tf
import numpy as np

IMAGE_SIZE = 512
anomolies = 6

#collect all images paths under data folder and store the path
IMG_Path = tf.data.Dataset.list_files("data/*.jpg", shuffle = False)
print(IMG_Path)

for x in IMG_Path:
    print(x)

    
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels =3)
    #image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)/255.0
    return image

