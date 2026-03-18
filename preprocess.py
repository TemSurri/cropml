import tensorflow as tf
import json
import os
from dotenv import load_dotenv

# Loading dotenv
load_dotenv()
dataset_root = os.getenv("DATASET_ROOT")
# Create Metadata list 

ANOMALY_NAMES = [
    "planter_skip",
    "water",
    "weed_cluster",
    "waterway",
    "nutrient_deficiency"
]

DATASET_ROOT = dataset_root

NUM_ANOMALIES = 5
IMAGE_SIZE = 512

# This function just builds the sample dictionary structure I layed out in the ReadMe
def build_sample_from_bounds_path(bounds_path):
    filename = os.path.basename(bounds_path)              
    stem = os.path.splitext(filename)[0]                

    sample = {
        "rgb": os.path.join(DATASET_ROOT, "field_images", "rgb", stem + ".jpg"),
        "nir": os.path.join(DATASET_ROOT, "field_images", "nir", stem + ".jpg"),
        "masks": {
            name: os.path.join(DATASET_ROOT, "field_labels", name, stem + ".png")
            for name in ANOMALY_NAMES
        },
        "valid_mask": os.path.join(DATASET_ROOT, "field_masks", stem + ".png")
    }
    return sample

def build_metadata_lists_from_json(json_path):
    with open(json_path, "r") as f:
        split_data = json.load(f)

    train_samples = []
    val_samples = []
    test_samples = []

    for split_name, split_entries in split_data.items():
        for bounds_path, metadata in split_entries.items():
            sample = build_sample_from_bounds_path(bounds_path)

            # extra metadata 
            sample["label_counts"] = metadata.get("label_counts", {})
            sample["label_areas"] = metadata.get("label_areas", {})
            sample["image_area"] = metadata.get("image_area", None)

            if split_name == "train":
                train_samples.append(sample)
            elif split_name == "val":
                val_samples.append(sample)
            elif split_name == "test":
                test_samples.append(sample)

    return train_samples, val_samples, test_samples


field_stats_path = "data/field_stats.json"
train_samples, val_samples, test_samples = build_metadata_lists_from_json(field_stats_path)

# Batch samples 
# Essentially this function just batches any list based on the batch_size arguement
def get_metadata_batches(samples, batch_size):
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    return batches


# Preprocess functions

#loads main image by combining the rgb and nir images as one
def load_image(rgb_path, nir_path):
    rgb_bytes = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_jpeg(rgb_bytes, channels=3)
    rgb = tf.cast(rgb, tf.float32) / 255.0
    rgb = tf.ensure_shape(rgb, [IMAGE_SIZE, IMAGE_SIZE, 3])

    nir_bytes = tf.io.read_file(nir_path)
    nir = tf.image.decode_jpeg(nir_bytes, channels=1)
    nir = tf.cast(nir, tf.float32) / 255.0
    nir = tf.ensure_shape(nir, [IMAGE_SIZE, IMAGE_SIZE, 1])

    image = tf.concat([rgb, nir], axis=-1)
    return image

#loads binary masks, image into tensor of 1's and 0's. 1 = true.
def load_binary_mask(mask_path):
    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask = tf.cast(mask > 0, tf.float32)
    mask = tf.ensure_shape(mask, [IMAGE_SIZE, IMAGE_SIZE, 1])
    return mask

# mask_paths must follow the same order as ANOMALY_NAMES
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

def preprocess_batch(batch_samples, training=True):
    images = []
    masks = []
    valid_masks = []

    for sample in batch_samples:
        mask_paths = [sample["masks"][name] for name in ANOMALY_NAMES]

        output = preprocess_sample(
            sample["rgb"],
            sample["nir"],
            mask_paths,
            sample["valid_mask"],
            training=training
        )

        image, mask, valid_mask = output
        images.append(image)
        masks.append(mask)
        valid_masks.append(valid_mask)

    batch_images = tf.stack(images, axis=0)
    batch_masks = tf.stack(masks, axis=0)
    batch_valid_masks = tf.stack(valid_masks, axis=0)

    return batch_images, batch_masks, batch_valid_masks




# Test batching + preprocessing flow

print("train:", len(train_samples))
print("val:", len(val_samples))
print("test:", len(test_samples))
print(train_samples[0])

BATCH_SIZE = 2

train_batches = get_metadata_batches(train_samples, BATCH_SIZE)

print("num train batches:", len(train_batches))
print("first batch size:", len(train_batches[0]))

batch_images, batch_masks, batch_valid_masks = preprocess_batch(
    train_batches[0],
    training=True
)

print("batch_images shape:", batch_images.shape)
print("batch_masks shape:", batch_masks.shape)
print("batch_valid_masks shape:", batch_valid_masks.shape)