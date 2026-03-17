First ML pipeline plan

Model Type : Classification & Segmentation

We will train a segmentation model with 5-8 output channels, one for each class/label. Since each output mask already indicates whether a label is present and where it is located, image-level classification can be derived from the predicted masks by checking whether each mask contains a sufficiently large positive region. This avoids adding a separate classification head in the first version of the pipeline.

TRAINING STEPS

1. 
Keep a list of meta data for samples
Each sample should contain paths for:
- the image
- the 5 anomaly masks
- boundary / valid mask if needed

Sample will be stored as a dictionary / hashmap:

{
    "rgb": ".../field_images/rgb/1C11HIBEV_....png",
    "nir": ".../field_images/nir/1C11HIBEV_....png",
    "masks": {
        "planter_skip": ".../field_labels/planter_skip/1C11HIBEV_....png",
        "water": ".../field_labels/water/1C11HIBEV_....png",
        "weed_cluster": ".../field_labels/weed_cluster/1C11HIBEV_....png",
        "waterway": ".../field_labels/waterway/1C11HIBEV_....png",
        "nutrient_deficiency": ".../field_labels/nutrient_deficiency/1C11HIBEV_....png"
    },
    "valid_mask": ".../field_masks/1C11HIBEV_....png"
}

2. Batch Metadata
We will manually batch metadata


3. Preprocess metadata per batch

Create a function A that for each image in a batch
- load images into matrices
- make sure images are of 512 x 512
- normalize entries
- augment the matrices(make more variation)
- keep the labels matched(answerkey) with the images
In this case each image will have 5 binary masks, one for each label we are testing.
- returns training images as matrixes corresponding to their answer(labelled) matrix, perhaps in a dictionary/hashmap.

TRAINING:
Loss function :
- we need a function that uses function A which outputs the training images all as one tensor, and runs it through the model, and compares the model output tensor with the asnwerkey tensor, and returns on scalar value(aka the loss).  

Iterate through the batches over epochs, in each iteration adjust the model weights using the loss function & other calcultions.

Other calculations include: 
- to do later ?????