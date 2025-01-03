# SimpleClassifier

This is the official GitHub repository for the paper: ["Simple is More: Efficient Liver View Classification in Ultrasound Images Using Minimal Labeled Data and Simple Neural Network Architecture"](https://link.springer.com/chapter/10.1007/978-3-031-73748-0_11), published at the Data Engineering in Medical Imaging (DEMI) workshop @ MICCAI 2024!

## Setup

The code in this repository is meant for performing binary classification, which involves distinguishing between two classes. In the context of the paper, the classification task focused on identifying whether a given ultrasound image contains the liver region or not. 

Assume your class names are `A` and `B`. In your main project folder, do the following:

1. Create a folder called `data` within your project directory.

2. Inside the `data` folder, create the following subfolders:
   - `unlabeled`: This folder will contain images without any assigned class labels.
   - `training`: This folder contains the images used for training the neural network.
   - `validation`: This folder contains the images used for neural network validation.
   - `test`: This folder contains the images used for testing the trained model.

3. Within the `training`, `validation`, and `test` subfolders, create two more subfolders for each class:
   - For `training` create folders named `A` and `B`.
   - For `validation` create folders named `A` and `B`.
   - For `test` create folders named `A` and `B`.

4. Place the images that belong to each class into their respective `A` and `B` folders within the `training`, `validation`, and `test` subfolders. Those images represent the labeled data.

5. The `unlabeled` folder should contain images that have not been assigned to any specific class.

Your folder structure should now look as follows:

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/project_folder.png" alt="project_folder_structure" width="500"/>
</div>

## 1. Uncertainty sampling

### 1.1. Train the binary classifier

Assuming you have images in the `validation` and `test` folders, you can begin by randomly selecting (and assigning labels to) two images from the `unlabeled` folder. At this stage, ensure that one image is assigned to class `A`, and the other image is assigned to class `B`.

Now, run the [`simple_classifier.py`](https://github.com/abderhasan/SimpleClassifier/blob/main/simple_classifier.py) script. Once the script finishes running, it will output the accuracy of the trained model when applied to the test dataset. Additionally, it will generate a file named `unlabeled_probabilities.csv`. This file contains the probability distribution for the unlabeled images located in your `unlabeled` folder. The format of the CSV file is as follows:

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/table.png" alt="table" width="250"/>
</div>

### 1.2. Find least confidence scores

To calculate the least confidence score from the probability distribution for each prediction, run the [`least_confidence_score.py`](https://github.com/abderhasan/SimpleClassifier/blob/main/least_confidence_score.py) script. The primary parameters required for this script are the input and output CSV files. In the script, these are defined as follows:

```python
input_csv_file = 'unlabeled_probabilities.csv'
output_csv_file = 'least_confidence_scores.csv'
```

The results in `least_confidence_scores.csv` will be sorted in ascending order. So, the content in the CSV file would look something like the following:

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/table_2.png" alt="table" width="500"/>
</div>

### 1.3. Sample data for domain expert labeling

Having calculated the least confidence scores, we will now select the top *N* samples that will be labeled by the domain expert (i.e., radiologist). To accomplish this, you can execute the [`select_data.py`](https://github.com/abderhasan/SimpleClassifier/blob/main/select_data.py) script; in this script, the first *50* samples are selected, but you can change the number of samples you would like to select.

### 1.4. Add new labeled data to the training dataset

In the previous step, you selected a sample of data and labeled it. These newly labeled images would now be added to your existing training dataset. Let's say you chose 50 images to label. If your training data initially contained *2* images, after adding the newly labeled images, your training dataset should now have a total of *52* images.

Once you've completed this step, go ahead and run the [`simple_classifier.py`](https://github.com/abderhasan/SimpleClassifier/blob/main/simple_classifier.py) script. This will train your model again, but this time using the updated training dataset. If the model's classification performance (i.e., test accuracy) still doesn't meet your expectations, you can go back and repeat steps 1.1 through 1.4.

## 2. Diversity sampling

### 2.1. Train the binary classifier

This is similar to step 1.1, except that you don't need the [`test_and_save_probabilities_unlabeled()`](https://github.com/abderhasan/SimpleClassifier/blob/main/simple_classifier.py#L125) function in `simple_classifier.py`. So, you can comment out the function from the classifier when working on diversity sampling.

### 2.2. K-means (cluster-based sampling)

To categorize unlabeled data into distinct groups, you can use the [`k_means.py`](https://github.com/abderhasan/SimpleClassifier/blob/main/k_means.py) script. In this script, you can specify the number of clusters you want, which is denoted as `k`. For example, in this script `k=5`, which will create 5 clusters. 

The script will generate *k* CSV files, named as `class_1.csv`, `class_2.csv`, ..., `class_5.csv`. Each of these CSV files contains information about the images assigned to its respective class. This information includes the image's name, whether it serves as the representative centroid image for that class, and if the image is identified as an outlier. The content in the CSV file would look something like the following:

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/k_means.png" alt="table" width="500"/>
</div>

To move the images listed in the CSV files to their respective folders (i.e., class_1, class_1_centroid, class_1_outliers), run [`csv_to_folders_diversity.py`](https://github.com/abderhasan/SimpleClassifier/blob/main/csv_to_folders_diversity.py).

### 2.3. Sample data for domain expert labeling

After identifying the classes and determining which images belong to each class, as well as finding the centroid images and outlier images for each class, the next step is to gather a sample of data for domain expert labeling. In the tutorial, this was done using the following procedure:

**2.3.1. Randomly select N images from each class folder**

Randomly choose *N* (i.e., *N=10*) images from each class folder.

**2.3.2. Select the centroid image of each class**

From the previously identified centroid images for each class, we include these as part of our sample data. These centroid images represent the representative image for each class.

**2.3.3. Select M outlier images of each class**

We also include a set number of outlier images from each class. In this case, *M* (i.e., *M=5*) outlier images per class are chosen. These outlier images are different from the typical images in the class and can help in identifying unusual or atypical cases.

To carry out the above steps, you can run this script: [`move_images_class_diversity.py`](https://github.com/abderhasan/SimpleClassifier/blob/main/move_images_class_diversity.py).

### 2.4. Add new labeled data to the training dataset

After you have collected and labeled your data, the next step is to incorporate this new data into your existing training dataset. Let's assume you had *5* classes and selected *10* random images, *1* centroid image, and *5* outlier images from each class. If your training data originally contained only *2* images, after adding these newly labeled images, your training dataset will now consist of a total of *82* images.

Once you have completed this data integration, proceed to run the [`simple_classifier.py`]([https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/simple_classifier.py](https://github.com/abderhasan/SimpleClassifier/blob/main/simple_classifier.py)) script. This script will retrain your model using the updated training dataset. If the model's classification performance (i.e. test accuracy) still does not meet your expectations, you can revisit and repeat steps 2.1 through 2.4 as needed.

