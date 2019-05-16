# -facerecognition

Web application writen in flask which can recognize face using convolutional siamese neural network.

Convolutional siamese neural network is actually made by using two identical networks (they have the same weights).

Usually, they are performing binary classification at the output, classifying if the inputs are of the same class or not. 

![123](https://user-images.githubusercontent.com/29351335/57818588-ce2db080-7784-11e9-9d38-f3babcee7d87.PNG)

Siamese networks have wide-ranging applications. One of the most popular ones is one-shot learning which is also used in my solution.

In this learning scenario, a new training dataset is presented to the trained (classification) network, with only one sample per class.
Afterwards, the classification performance on this new dataset is tested on a separate testing dataset.
As siamese networks first learn discriminative features for a large specific dataset, they can be used to generalize this knowledge to entirely new classes and distributions as well. 


To prepare network you need to find proper dataset. The best which i found can be downloaded here:(http://www.vap.aau.dk/rgb-d-face-database/)

Even though this dataset has its drawbacks i think that for this network it is good enough. Before using it i recommend grouping pictures and resieing them ( i used (200,200,3) size).

To have accurate network weights i used the ones mentioned here (http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) 

The rest is just preparing pairs of pictures, some from the same category (the same person is there) and some from 2 different categories.
