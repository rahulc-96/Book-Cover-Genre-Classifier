
<h1>A Classifier for Book Cover Genre’s using CNN machine learning model.</h1>



<h2>Google Colab</h2>

Google Colaboratory is a free online cloud-based Jupyter notebook environment that allows us to train our machine learning and deep learning models. All that you need to set up a google colab is a google user account and a web browser, colab lets you leverage the free GPU and TPU instances provided by google to train complex machine learning models in matter of minutes that would have otherwise taken hours on a CPU. 

Colab notebooks have most of the common python libraries installed on them by default. If any additional libraries are required we can install them on our notebook as colab supports regular terminal commands, all that you have do is add an exclamation mark “!” to the beginning of each command. e.g: <b><i>!pip install library_name</i></b>

Colab notebook can be executed continuously for 12 hours after which the memory will be cleared. However, we can mount our google drive onto the colab instance and use the drive as location for saving computed models or the datasets that we have fetched from sources like Kaggle or uploaded from our local machine.


<h2>TensorFlow’s Keras API</h2>

It’s a high-level API for building and training machine learning models. The key advantages of using Keras is that it provides a simplified approach for building up models by connecting multiple individual blocks each block customizable with few restrictions. Keras offers a variety of pre-trained models like Resnet that have been trained on large Image datasets and can be used to tackle Image Classification ML problems by using these learned models than training a model from the scratch on a large dataset.  Keras also provides good support for diagnosing the performance and accuracy of our trained models.










<h2>Implementation</h2>

<h3>Loading Image files from the dataset as NumPy arrays</h3>

<b>Book Cover Dataset</b> - The dataset contains around 33 different book categories and each category has around 1000 book covers in JPEG format. Each book cover belongs to only one category. Kaggle provides python libraries that can be installed on the colab notebook and the dataset can be loaded using Kaggle API.

For training the model we have chosen 5 different book categories. Each book cover is loaded from the file system in jpeg format, converted to arrays of shape 224*224*3 (width x height x RGB channels) and appended to a ‘images’ list. The ‘labels’ list contains integers, with the value at each index representing the category of book-cover at the corresponding index in the ‘images’ list. The images are selected randomly from the five chosen categories, this is to ensure that while training the model in batches, each batch has relatively equal representation of book-covers from all the 5 categories, this prevents the model from being biased to one category. The loaded images and labels are then saved to the file system as npz files.


<h3>Convolutions Neural Networks (CNN)</h3> 

Convolution neural network is a deep learning algorithm that has been successfully employed for analyzing visual imagery. The working of a CNN in simple terms - the image is broken down into a number of tiles and the machine tries to predict what each tile is. Finally, the machine evaluates what the image is based on the prediction of all these small tiles.

A CNN is composed of three layers:
<ol>
<li>Convolutional layer</li>
<li>Max Pooling layer</li>
<li>Fully Connected layer.</li></ol>

![CNN](https://user-images.githubusercontent.com/29629955/84566056-b95a8c80-ad8b-11ea-9c43-0f37663d5dc7.jpg)

<b>Convolutional Layer</b> - The convolution operation is performed to extract the high level featured such as edges and colors. Adding more convolutional layers to the CNN will help the model to extract complex features and have a wholesome understanding of images in the dataset.


<b>Max Pooling Layer</b> - The max pooling operation aids in decreasing the computational power required to process the data by applying dimensionality reduction and also help in suppressing noise within the data.


<b>Fully Connected Layer</b> - The convolutional and max pooling layers mainly contribute to the understanding capability of the model. The fully connected layer is the one that handles the classification part of the model. The term “Fully Connected” implies that every neuron in the previous layer is connected to every neuron on the next layer. The features extracted from the previous layers are flattened to an output column vector and passed to SoftMax Activation Function to generate the final output. The final output is a vector of length equal to the number of classes, in our example it’s the number of different book categories, with the value at each index representing the probability that the input falls into this class.


<b>Why SoftMax Function is used as the output layer for classification?</b>
The SoftMax function takes a vector of arbitrary real-valued scores and squashes it to a vector of values between zero and one that sum to one.








<h3>Data-Preprocessing</h3>

Some of the data pre-processing techniques that has been applied to the data before feeding into the model for training:
<ul>
<li>The images are loaded as squares of size 224 * 224 this ensures all the images used for training as a uniform shape.</li>

<li>For each image the pixel value is normalized to a range between 0 and 1. The objective of normalization is to have a uniform distribution for pixel values which will simplify the training process.</li>

<li>Data Augmentation: Augmenting the dataset with altered versions of the existing images by introducing horizontal or vertical flips and shifts. This step ensures that the model is exposed to a wide variety of variations and learns features irrespective of position.</li>
</ul>

<h3>Concept of Transfer Learning</h3>

Transfer learning refers to the concept of reusing a model trained on one problem to solve a similar problem. The benefits of transfer learning include lesser training time and lower generalization error. Keras API provides a wide range of pre-trained models for image classification problems like the Resnet and VGG Models. These models have been trained on millions of images and are very effective in detecting generic features from images.

The most common usage patterns are as follows:

<ul>
<li>Classifier: the pre-trained model is used to directly classify the images.</li>
<li>Feature-extraction: the pre-trained model as a whole or a part of it is used to extract features and passed on to a new model.</li>
<li>Weight Initialization: the pre-trained model is integrated with a new custom model and the layers of the pre-trained model is trained along with that of the new model.</li>
</ul>
For the given book cover classification problem, we have used a Resnet50 pre-trained convolutional neural network that has 50 layers. The extracted features from the Resnet are fed to a fully connected layer that generates the final classification result for the input image.





<h3>Model Diagnosis</h3>

The ideal loss function to be used for a multi class classification problem is <b>the cross-entropy loss function</b>. Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for all classes in the problem. The score is minimized and a perfect cross-entropy value is 0.

Graphs are plotted for accuracy and loss against epochs used for training the model. These graphs are used for analyzing the model’s performance and how the parameters can be tweaked to tackle problems of underfitting or overfitting data.
![diagonosis](https://user-images.githubusercontent.com/29629955/84566151-b01def80-ad8c-11ea-94dc-2aa1d241c49f.jpg)


  

              
 
    






<h3>Loading the computed model and using the model to classify new book-covers</h3>

Once the CNN model has been trained over the entire data set, the computer model is saved to the system as a h5 file. A simple python script can be used to load the computed model and on passing a book cover as input image the model will predict the percentage probability for the cover to belong to each of the five categories under consideration.

![output](https://user-images.githubusercontent.com/29629955/84566209-1571e080-ad8d-11ea-968b-014a8f259cb4.jpg)




<h4>For further reading</h4>
Source Colab Notebook : https://colab.research.google.com/drive/13ih4o1VsTDTuJkOT7JJk0oEiXShFE37e?usp=sharing
Introduction to Google Colab: https://colab.research.google.com/notebooks/intro.ipynb
<br>
Keras API: https://keras.io/api/
<br>
Google Colab Notebook for book classifier code: https://colab.research.google.com/drive/13ih4o1VsTDTuJkOT7JJk0oEiXShFE37e?usp=sharing


