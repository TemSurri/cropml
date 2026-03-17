import tensorflow as tf
import numpy as np

NUM_ANOMALIES = 5
IMAGE_SIZE = 512

ANOMALY_NAMES = [
    "planter_skip",
    "water",
    "weed_cluster",
    "waterway",
    "nutrient_deficiency"
]

# need the training image paths stored as in the readme
dataset = []








def load_image(rgb_path, nir_path):
    rgb_bytes = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_png(rgb_bytes, channels=3)
    rgb = tf.cast(rgb, tf.float32) / 255.0
    rgb = tf.ensure_shape(rgb, [IMAGE_SIZE, IMAGE_SIZE, 3])

    nir_bytes = tf.io.read_file(nir_path)
    nir = tf.image.decode_png(nir_bytes, channels=1)
    nir = tf.cast(nir, tf.float32) / 255.0
    nir = tf.ensure_shape(nir, [IMAGE_SIZE, IMAGE_SIZE, 1])

    image = tf.concat([rgb, nir], axis=-1)   # shape: (512, 512, 4)
    return image

def load_binary_mask(mask_path):
    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask = tf.cast(mask > 0, tf.float32)
    mask = tf.ensure_shape(mask, [IMAGE_SIZE, IMAGE_SIZE, 1])
    return mask

def load_all_masks(mask_paths):
    masks = [load_binary_mask(path) for path in mask_paths]
    masks = tf.concat(masks, axis=-1)   # shape: (512, 512, 5)
    return masks

def load_valid_mask(valid_mask_path):
    mask_bytes = tf.io.read_file(valid_mask_path)
    valid_mask = tf.image.decode_png(mask_bytes, channels=1)
    valid_mask = tf.cast(valid_mask > 0, tf.float32)
    valid_mask = tf.ensure_shape(valid_mask, [IMAGE_SIZE, IMAGE_SIZE, 1])
    return valid_mask

#augmentation adds more variability to the training data
def augment(image, masks, valid_mask=None):
    do_hflip = tf.random.uniform(()) > 0.5
    do_vflip = tf.random.uniform(()) > 0.5

    image = tf.cond(do_hflip, lambda: tf.image.flip_left_right(image), lambda: image)
    masks = tf.cond(do_hflip, lambda: tf.image.flip_left_right(masks), lambda: masks)

    image = tf.cond(do_vflip, lambda: tf.image.flip_up_down(image), lambda: image)
    masks = tf.cond(do_vflip, lambda: tf.image.flip_up_down(masks), lambda: masks)

    if valid_mask is not None:
        valid_mask = tf.cond(do_hflip, lambda: tf.image.flip_left_right(valid_mask), lambda: valid_mask)
        valid_mask = tf.cond(do_vflip, lambda: tf.image.flip_up_down(valid_mask), lambda: valid_mask)

    return image, masks, valid_mask

#complete preprocess 
def preprocess_sample(rgb_path, nir_path, mask_paths, valid_mask_path=None, training=True):
    image = load_image(rgb_path, nir_path)
    masks = load_all_masks(mask_paths)

    valid_mask = None
    if valid_mask_path:
        valid_mask = load_valid_mask(valid_mask_path)

    if training:
        image, masks, valid_mask = augment(image, masks, valid_mask)

    if valid_mask is not None:
        return image, masks, valid_mask
    else:
        return image, masks

