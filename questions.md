# Week 1

### **Questions** 
  - **Question 1:** When we pretend to identify the artist and year to which the painting corresponds, it will be necessary to use more than one network, that is, one to detect the artist and another to detect the year or it can be done all in one raising a Multi-label problem?
  - **Question 2:** Are there biases in the data, i.e. is the amount of paintings equal for each artist/year?
  - **Question 3:** What results does the first starting point for classification show? How does its code work? Is the model used suitable for our data?

### **Answers**

  - **Answer 1:**

As our intention is to identify the chronological career of artists, we should first detect which artist each painting belongs to. To achieve this, we need to create a classifier that can identify the artist of each painting. Once this is done, when the artist is correctly identified, we should then arrange each of their paintings in chronological order. Therefore, these are two distinct tasks that cannot be done simultaneously; rather, we must first classify each painting by artist and then sort the paintings by date.

  - **Answer 2:**

Considering the provided data, we observe that the distribution of artists (Image 1) appears to be more balanced compared to the distribution of dates (Image 2). There are a total of 2319 distinct artists and 1690 distinct dates. Notably, there are 12 artists that appear exactly 500 times.

Regarding biases in the data, it's evident that the amount of paintings is not equal for each artist or year. Some artists may have a higher number of paintings compared to others, and similarly, some years may have more paintings represented than others. This suggests potential biases in the dataset, indicating that certain artists or time periods might be overrepresented or underrepresented compared to others.

However, in our case, we will always work with the top n artists that appear most frequently in the dataset, making the bias towards artists less significant initially.

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/edd6a54d-861b-432a-be24-d5c653a0a63f" alt="Descripción de la imagen">
  <br>
  Image 1. Artists Distribution
</p>


<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/b19c0053-a9f0-41e1-a8e3-29d59dbfeb29" alt="Descripción de la imagen">
  <br>
  Image 2. Date Distribution
</p>


  - **Answer 3:**

The classification model of the first starting point (https://www.kaggle.com/code/awallst/somethingaboutimages-csc-480-cal-poly) uses the ResNet50 classifier and initializes the weights with a file saved in hexadecimal format. It is important to note that the result for the "Top 5 artists with 100 images" dataset is not very good, showing a final accuracy on the validation set of 17%-20%. From epoch 8 onward, the validation accuracy does not improve, and the training accuracy remains at 100%, which suggests that the poor results may be due to the limitations of the dataset.

It remains pending to run this code on a larger dataset and to seek out other reference starting points to build another classifier. We believe there may be better versions using different models and following other techniques.

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/45a485b8-e7aa-4c63-9e16-1e13a4e0ecca">
  <br>
  Image 3. Accuracy results of TOP_5_ARTISTS_WITH_100_IMAGES (SP 1)
</p>

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/ad69e982-26d5-46a6-8197-776120c8f312">
  <br>
  Image 4. Loss results of TOP_5_ARTISTS_WITH_100_IMAGES (SP 1)
</p>



### **Summary of the week**

First of all, we started searching for diverse starting points in which we could begin to think about how our problem could be approached. The main problem we had at the beginning was that the full dataset took up too much to work in the azure host, so we decided to clean up locally the dataset and just take the images from autors that represent a specific percentage (for example, 8%, 20%)  of the total samples. To undestand better how we managed the storage problem you can take a look at the *Datasets_Creator.py* of the repository. Besides, we wanted to study if the dataset we are working with was biased in some way or not, so that we analized all the data by genres, autors and years. This lasts two are the main thing we are going to attempt to solve with our model, trying to explain the career of every artist by classifying properly their paintings and then arrenging the whole paintings in chronological order. In the *Data Distribution* folder, you will find the code used to perform this analysis. After the follow-up session, we settled on the main objective for next week: to try to improve the classifier we have so far and test other models to find the one that best suits our data.. 

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/131007325/197c3f89-dfe8-4e12-b24c-00bb63c1b234">
  <br>
  Image 5. Scheme implemented so far
</p>

# Week 2

### **Questions** 
  - **Question 1:** Which classification models/techniques have been tested?
  - **Question 2:** What results do each of the implemented models yield? Which model provides the best results across different datasets? What are the best configured hyperparameters in the implemented model(s) (learning rate, optimizers...)?
  - **Question 3:** How can we tackle the task of chronologically ordering the images of each artist?

### **Answers**
  - **Answer 1:**

Through a new Starting Point that has been found (https://www.kaggle.com/code/shtrausslearning/pytorch-multiclass-image-classification), various executions have been carried out for the *ResNet50*, *ResNet18*, and *Vgg16* classification models. Additionally, an attempt has been made to perform classification using a *Siamese Network* to employ a *metric learning* technique. While all of this has been happening, comparisons have been made with the results obtained to ultimately decide which model yields the best results using our data.

  - **Answer 2:**

What we have done is test different classifiers with various combinations of optimizers, learning rate values, and learning rate schedulers. With this approach, we aim to evaluate and compare different models to identify the best one for classifying the images in our dataset by artist. The models used are ResNet18, ResNet50, and VGG16.

Initially, two versions of each model were run: one trained from scratch and one pretrained (parameter pretrained = True). Setting the parameter pretrained = True means that the model uses weights from a network that has been previously trained on a large dataset, typically ImageNet. It was clearly observed that the pretrained models yielded better results. From this point onward, we worked exclusively with the pretrained models.

Subsequently, each of these models was first executed with the same learning rate and learning rate scheduler, varying only the optimizers. This allowed us to determine the best optimizer for each model. The optimizers used were:

  - **Adam:** Combines the advantages of two other extensions of stochastic gradient descent, AdaGrad and RMSProp. It works well in practice and is robust to noise.
  - **AdaGrad:** Adapts the learning rate to the parameters, performing larger updates for infrequent parameters and smaller updates for frequent ones. It is well-suited for sparse data.
  - **SGD (Stochastic Gradient Descent):** Updates the weights in the direction of the gradient of the loss function. It's simple and effective, but can be slow to converge.
  - **RMSprop:** Maintains a moving average of the square of gradients to adapt the learning rate for each parameter. It helps to stabilize the training process.

Once the best optimizer was identified, we conducted different runs using this optimizer but varied the learning rate scheduler. The different learning rate schedulers used were:

  - **StepLR:** Decays the learning rate by a factor every few epochs, which helps to gradually reduce the learning rate.
  - **MultiStepLR:** Allows more flexibility by specifying epochs at which to decay the learning rate by a factor.
  - **ExponentialLR:** Decays the learning rate exponentially based on the epoch number, providing a smooth transition.
  - **CosineAnnealingLR:** Adjusts the learning rate following a cosine curve, starting with a high learning rate and gradually reducing it, then increasing it again.

This methodology allowed us to identify the most effective combination of optimizer and learning rate scheduler for each model. Once this was done, we compared the best version of each model to finally determine the optimal version for classifying images by artist according to our data. It is important to note that all these executions were performed using the Top 5 artists with 100 images dataset.

Below are the graphs of the best classifier obtained for each model. It's important to note that overfitting is present in these and all other models tested. Now that we have identified the best model, our next step is to improve it to reduce this overfitting. This model comparison has helped us determine the best classifier for artists in our specific case.

**ResNet18:**
  - learning rate: 1e-4
  - Optimizer: SGD
  - lr Scheduler: MultiStepLR

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/76558feb-dc26-49e9-9806-884099498d4c">
  <br>
  Image 6. Results of ResNet18 best Model with the dataset "Top 5 artists with 100 images"
</p>

  **ResNet50:**
  - learning rate: 1e-4
  - Optimizer: AdaGrad
  - lr Scheduler: ExponentialLR

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/c3d64a9f-5042-4389-b442-2baf4ab8581f">
  <br>
  Image 7. Results of ResNet50 best Model with the dataset "Top 5 artists with 100 images"
</p>

**Vgg19**:
In the case of this model, we would like to highlight two runs. The first one shows higher accuracy but much more overfitting, as the validation loss increases significantly after epoch 15.

  **Vgg16_1:**

  - learning rate: 1e-4
  - Optimizer: Adam
  - lr Scheduler: ExponentialLR

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/c522ad22-2f30-4142-9bfa-2b17061e633b">
  <br>
  Image 8. Results of Vgg16 best Model 1 with the dataset "Top 5 artists with 100 images"
</p>

  **Vgg16_2:**
  - learning rate: 1e-4
  - Optimizer: Adam
  - lr Scheduler: MultipleLR

<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/4ba86fd8-1e5e-4dde-820b-a2d31ba8b773">
  <br>
  Image 9. Results of Vgg16 best Model 2 with the dataset "Top 5 artists with 100 images"
</p>


  - **Answer 3:**

Our idea to tackle this problem is to create a classifier that allows us to compare pairs of images and determine which one is older (in terms of date). By repeatedly applying this classifier, we can ultimately achieve the chronological ordering of an artist's images. This solution will then be compared with a metric learning algorithm, which aims to order the images chronologically in a spatial manner.

### **Summary of the week**

The main objective for this second week was to develop a good enough artists classifier. As we show in readme documentation we tried to approach the problem with 3 different models: ResNet18, ResNet50 and VGG16. For all of them we have done tests with pretrained and not pretrained versions (all the cases were better the pretrained ones) and an analysis of the results with different hyperparameters such as optimisers (Adam, AdaGrad, SGD, RMSProp), scheduler learning rates (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR). Basically what we could identify with tests is the most effective combination of parameters for each model for later comparision. The main problem we had is that every model showed a big overfitting and that's precisely what we've been trying to reduce in recent days by freezing latest layers of the models, transforming data and introducing dropout. Apart of this, note that we have started to perform metric learning to chronologically order the paintings.



<p align="center">
  <img src= "https://github.com/DCC-UAB/XNAPproject-grup_12/assets/131007325/6abf63a8-75ae-485c-8f0a-2ac30d00b723">
  <br>
  Image 10. Schema implemented so far
</p>


# Week 3
### **Questions** 
  - **Question 1:** What techniques have been developed to reduce overfitting in the artist classifier?
  - **Question 2:** Regarding ordering by date, what results does the classifier yield when determining which image is older out of two images?
  - **Question 3:** Regarding ordering by date, what are the initial results provided by the implemented metric learning?

### **Answers**
  - **Answer 1:** For the moment, aside from experimenting with different optimizers and learning rate schedulers, we have had to change our training approach by using layer freezing and unfreezing since the differences with these methods were not significant enough in reducing overfitting. Thus, we have been implementing this both in the initial layers and in the final layers, including the fully connected layer, and in some cases, the model's performance has been much more effective. It is important to note that all tests have been initially conducted on the TOP100_ARTISTS_WITH_ALL_PICTURES dataset. 
  - **Answer 2:** In terms of ordering by date using a binary classifier, we developed two models. Firstly, we constructed 
a VGG16 network that compares two photos and then determines which one is older. Subsequently, employing the same format, we implemented a ResNet18 to test a simpler network.

For VGG, we tried different ways of training the model. Firstly, we created training pairs composed of works from the same artist. However, this type of training caused our model to overfit. Then, we tried using pairs composed of works by different artists, and this was the result.

![image](https://github.com/DCC-UAB/XNAPproject-grup_12/assets/104071433/7f267da6-a80e-471d-97d3-a7bd897f5d8e)

    

  - VGG16: Initially, we built a model using VGG16 architecture, incorporating early stopping to mitigate overfitting. This was the outcome using a dataset of 15 dates with 100 pictures for each date.

<p align="center">
  <img src= "https://github.com/DCC-UAB/XNAPproject-grup_12/assets/104071433/3688f4ee-aa9a-41a8-8ade-b01688d9d61e">
  <br>
  Image 11. VGG16 results
</p>
    As we can see in the picture, this model have a lot of overfitting so we decided to make a larger execution using the all images of 100 artists.

  - **Answer 3:** First, we focused our work on trying to create a class capable of generating triplets considering the difference in years between pairs of images. To assign an image as positive relative to its anchor, we used a year threshold. For example, if the threshold is 5, the positive component of the triplet relative to the anchor can be from the same year or up to five years older or younger. Once we created the triplets, we aimed to develop a model to extract features from the images, then calculate embeddings, and finally predict and evaluate.
The initial results were very poor, with approximately 6% accuracy. Next, we attempted to create an ensemble of pretrained models (ResNet121, DenseNet, MobileNet) that works in such a way that each of the three models is trained individually. We then stack the feature vectors and create combined embeddings, which are subsequently used to train a meta-learner (using SVR, RandomForest, or LinearRegressor). Once trained, the meta-learner makes predictions on the test data, and we evaluate the experiments using accuracy and MSE.


### **Summary of the week**
This third week, on one hand, for date ordering, we developed binary classifiers with VGG16 and ResNet18, initially facing overfitting issues like we had done before. On the other hand, metric learning with triplets yielded poor results at the beginning, so we shifted to an ensemble approach using pretrained models and meta-learners (machine learning estimators like SVR, RF, GradientBoosting), showing promise with combined embeddings. We observed that the predictions followed a distribution close to the mean in the distribution of dates, so we decided to retry the previous metric learning approach, but this time customizing the data in each batch by taking just images from an unique artist in every batch. In order to evaluate the performance of the model we used the metrics of _r_precision_, _precision_at_1_, _precision_at_3_ and _mean_average_precision_. Results still weren't as good as we expected, so our thoughts were that with our data we can't order like this the images of all the artists, but we decided to train and test images only from a dataset of images from one of the artists. The chosen one has been Vincent Van Gogh, and the model perform much better than before reaching almost 0.60 at precision_at_1 metric.

<p align="center">
  <img src= "https://github.com/DCC-UAB/XNAPproject-grup_12/assets/127510405/61b013d5-7ac9-49ab-b106-aa84ca0fb18a">
  <br>
  Image 12. Schema implemented so far
</p>

