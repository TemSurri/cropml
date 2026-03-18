First ML pipeline plan

Model Type : Classification & Segmentation

We will train a segmentation model with 5-8 output channels, one for each class/label. Since each output mask already indicates whether a label is present and where it is located, image-level classification can be derived from the predicted masks by checking whether each mask contains a sufficiently large positive region. This avoids adding a separate classification head in the first version of the pipeline.

Preprocessing -------------------------------------------------------------------------------------------------
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

Training ---------------------------------------------------------------------------------:
Now that we have the data ready, we must:
- Define the built in model to use
- Configure the model to predict : needs to take in our input and give expected output.
- Make loss function : see how wrong the prediction is
- GRADIENTS?
- update model weights


1. Defining model
- defines the neural network architecture ( we can pick any from  {simple CNN, U-Net, DeepLab} )
For our purposes I think we should use U-net.
- how features are extracted and how output masks are made
For our case : input shape (512,512,4) | output shape (512,512,5)


2. Loss function (Prediction, Answer) :
- Measures how wrong the models output is. Arguements should be Prediction and AnswerKey. possibly valid mask as well.
Theres many different built in loss functions we can probably use.

3. Gradients :
- tells us how much each parameter in the model contributes to the loss function
- gradients are essentially are how much the loss function is effected based on a weight/paramter.
- These gradients are then passed into the optimizer function.

5. Optimizer Function :
- Evaluates how the weights of the model should be changed to optimize for a lower yield for the loss function
Again there are many build in optimzer functions we could probably use.

5. train_step :
A function that does the entire training process over one batch : to be used in the main pipeline

6. val_step:
Same as train_step but for the testing data

  
Other calculations include: 
- to do later ?????
