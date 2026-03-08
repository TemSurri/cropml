First ML pipeline plan

Model Type : Classification & Segmentation

We will train a segmentation model with 6 output channels, one for each class/label. Since each output mask already indicates whether a label is present and where it is located, image-level classification can be derived from the predicted masks by checking whether each mask contains a sufficiently large positive region. This avoids adding a separate classification head in the first version of the pipeline.

TRAINING STEPS

Preprocessing Data:
- have all the data file names organized in a list/ or other convenient data strucutre
- batch the data & organize accordingly to training and testing data.

Create a function A that for each image in a batch
- load images into matrices
- make sure images are of 512 x 512
- normalize entries
- augment the matrices(make more variation, will cause the function to return more images that input)
- keep the labels matched(answerkey) with the images
In this case each image will have 6 binary masks, one for each label we are testing.
- returns training images as matrixes corresponding to their answer(labelled) matrix, perhaps in a dictionary/hashmap.

TRAINING:
Loss function :
- we need a function that uses function A which outputs the training images all as one tensor, and runs it through the model, and compares the model output tensor with the asnwerkey tensor, or iterate through the tensor that was outputted from the model and the asnerkey tensor and record the error for each training image. Then convert all that to one number via an average or other statistical calcualtion.  
