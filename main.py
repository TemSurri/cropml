from preprocess import train_samples, val_samples, get_metadata_batches, preprocess_batch
from training import (
    build_simple_segmentation_model,
    create_optimizer,
    train_step,
    val_step
)

BATCH_SIZE = 2
EPOCHS = 5

# build metadata batches
train_samples_small = train_samples[:32]
val_samples_small = val_samples[:16]

train_batches = get_metadata_batches(train_samples_small, BATCH_SIZE)
val_batches = get_metadata_batches(val_samples_small, BATCH_SIZE)

model = build_simple_segmentation_model()
optimizer = create_optimizer(learning_rate = 1e-4)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_losses = []

    for batch in train_batches:
        batch_images, batch_masks, batch_valid_masks = preprocess_batch(
            batch,
            training=True
        )

        train_loss = train_step(
            model,
            optimizer,
            batch_images,
            batch_masks,
            batch_valid_masks
        )

        train_losses.append(float(train_loss))
        print("train loss:", float(train_loss))

    avg_train_loss = sum(train_losses) / len(train_losses)

    val_losses = []

    for batch in val_batches:
        batch_images, batch_masks, batch_valid_masks = preprocess_batch(
            batch,
            training=False
        )

        val_loss = val_step(
            model,
            batch_images,
            batch_masks,
            batch_valid_masks
        )

        val_losses.append(float(val_loss))

    avg_val_loss = sum(val_losses) / len(val_losses)

    print("avg train loss:", avg_train_loss)
    print("avg val loss:", avg_val_loss)



