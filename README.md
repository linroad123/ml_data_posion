# Machine Learning -- Evasion Attacks


## Background

If the machine learning model is trained based on data from potentially untrustworthy sources (such as Yelp, Twitter, etc.), an attacker can easily manipulate the training data distribution by inserting elaborate samples into the training set to change the model's behavior and reduce the mode's performance.

This type of attack is called "Data Poisoning" attack. It has not only received widespread attention in academia, but also caused serious harm in industry. For example, Microsoft Tay, a chatbot designed to talk to Twitter users, was shut down only 16 hours later because it started to make racist-related comments after being attacked by poisoning. This attack makes us have to rethink the security of machine learning models.

## Usage

### 1.py
![image](http://github.com/itmyhome2013/readme_add_pic/raw/master/images/nongshalie.jpg)

It can be seen that the contour map of the decision function of the MLP classifier basically fits our data set (the red dots are basically in the red shaded part, and the blue points are basically in the blue shaded part)

Here we use a confidence threshold of 0.5 as the decision boundary, that is, if the classifier predicts P(y=1)>0.5, then predict y=1; otherwise predict y=0


### 2.py
![image](http://github.com/itmyhome2013/readme_add_pic/raw/master/images/nongshalie.jpg)

### add.py

![image](http://github.com/itmyhome2013/readme_add_pic/raw/master/images/nongshalie.jpg)

It can be seen that the current 5 points (represented by a five-pointed star) are added in the space of y=1 (y=1 is represented by a hollow circle, and y=0 is represented by a solid circle)

### 3.py
Data poisoning
To simulate attackers dynamically and incrementally attacking machine learning models, we use scikit-learning's partial_fit() API for incremental learning. We incrementally train the existing classifier by fitting the model part to our newly added 5 points.

![image](http://github.com/itmyhome2013/readme_add_pic/raw/master/images/nongshalie.jpg)

At this time, there is an additional gray line, which is the new decision boundary, and we noticed that there has been an offset at this time. Through this shift, the part between the two decision-making boundaries that should have been classified as y=0 will now be classified as y=1.

It shows that the attacker has successfully caused the sample to be misclassified.

### 4.py
Repeat 5 times

![image](http://github.com/itmyhome2013/readme_add_pic/raw/master/images/nongshalie.jpg)

It can be seen that as partial_fit() is repeatedly used iteratively, the new boundary shifts more and more

### 5.py
Repeat 15 times directly

![image](http://github.com/itmyhome2013/readme_add_pic/raw/master/images/nongshalie.jpg)

The point pointed to by the red arrow in the above figure should be classified as y=1 under the original decision boundary, but under the current decision boundary, it should be classified as y=0. In this way, the attacker successfully attacked the machine learning model through data poisoning.

