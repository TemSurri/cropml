import tensorflow as tf

NUM_ANOMALIES = 5
IMAGE_SIZE = 512


def build_simple_segmentation_model():
    """
    Small starter segmentation model.

    Input:  (512, 512, 4)
    Output: (512, 512, 5)

    This is not meant to be the final architecture. We can decide on it in the lab.
    It is a lightweight model for validating the training pipeline.
    """
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 4))

    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)

    outputs = tf.keras.layers.Conv2D(
        NUM_ANOMALIES,
        kernel_size=1,
        padding="same",
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#this is the loss function using BCE
def masked_bce_loss(y_true, y_pred, valid_mask):
    """
    Computes Binary Cross Entropy loss, masked so that only valid pixels contribute.

    Args:
        y_true: ground truth masks, shape (batch, 512, 512, 5)
        y_pred: predicted masks,   shape (batch, 512, 512, 5)
        valid_mask: valid-region mask, shape (batch, 512, 512, 1)

    Returns:
        scalar loss
    """
    # BCE per pixel, per channel
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    # shape: (batch, 512, 512, 5)

    # expand valid mask over channels so it matches prediction channels
    valid_mask = tf.cast(valid_mask, tf.float32)
    valid_mask = tf.broadcast_to(valid_mask, tf.shape(bce))

    masked_loss = bce * valid_mask

    # normalize by number of valid entries
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(valid_mask) + 1e-8)


def create_optimizer(learning_rate=1e-4):
    """
    Creates the optimizer used during training. The return optimizer will be used to apply gradients later.
    """
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


@tf.function
def train_step(model, optimizer, batch_images, batch_masks, batch_valid_masks):
    """
    Runs one training step on one batch.

    Steps:
    1. forward pass
    2. compute loss
    3. compute gradients
    4. update weights
    """
    with tf.GradientTape() as tape:
        predictions = model(batch_images, training=True)
        loss = masked_bce_loss(batch_masks, predictions, batch_valid_masks)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def val_step(model, batch_images, batch_masks, batch_valid_masks):
    """
    Runs one validation step on one batch.

    No gradients are computed and no weights are updated.
    """
    predictions = model(batch_images, training=False)
    loss = masked_bce_loss(batch_masks, predictions, batch_valid_masks)
    return loss


