from preprocess import train_samples, val_samples, get_metadata_batches, preprocess_batch
from training import (
    build_simple_segmentation_model,
    create_optimizer,
    train_step,
    val_step
)

BATCH_SIZE = 2

# build metadata batches
train_batches = get_metadata_batches(train_samples, BATCH_SIZE)
val_batches = get_metadata_batches(val_samples, BATCH_SIZE)

# take just one batch
train_batch_samples = train_batches[0]
val_batch_samples = val_batches[0]

# preprocess one batch
batch_images, batch_masks, batch_valid_masks = preprocess_batch(
    train_batch_samples,
    training=True
)

# build model + optimizer
model = build_simple_segmentation_model()
optimizer = create_optimizer(learning_rate=1e-4)

# sanity check model output shape
predictions = model(batch_images, training=False)
print("predictions shape:", predictions.shape)

# one training step
train_loss = train_step(
    model,
    optimizer,
    batch_images,
    batch_masks,
    batch_valid_masks
)

print("train loss:", float(train_loss))

# one validation batch
val_images, val_masks, val_valid_masks = preprocess_batch(
    val_batch_samples,
    training=False
)

val_loss = val_step(
    model,
    val_images,
    val_masks,
    val_valid_masks
)

print("val loss:", float(val_loss))