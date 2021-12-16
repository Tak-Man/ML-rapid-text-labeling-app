# ML-rapid-text-labeling-app
Using machine-learning to enable a user to rapidly label a large text corpus.

This repo was originally produce for the final Capstone project for the University of 
Michigan Masters in Applied Data Science. It contains code for a working 
web app that implements a variety of Data Science techniques to enable a 
user to rapidly label a text corpus and have control over how they manage
 the trade-off between accuracy and speed in the labeling process.
 
 During development of the app the ['Disaster Tweets Dataset'](https://crisisnlp.qcri.org/humaid_dataset.html#) was used.
  The web app performed well in terms of speed and accuracy on this dataset.
  
In particular, the web app resulted in an 82% time saving to achieve the same accuracy threshold by 
smart selection of which texts to label next compared to labeling texts in a random order. The red bar  
shows how long it takes for accuracy to reach a key threshold where labels are added in a random order. 
The green bars show how quickly that threshold can be reached when the functionality of the app is used 
to make the labeling process smarter. This is done by using the ML model in the app to indicate which 
texts it is least confident about and getting the user to label those first.

![time saving at 0.95 threshold using web app difficult texts functionality](https://github.com/Tak-Man/ML-rapid-text-labeling/blob/main/web-testing/viz/time_saving_0.95.png)
 
It should be noted that in the intended real-world usage, the user would not
have labeled data to work with, so one of the **innovations** of this app is
an **in-app guide** of how well the user is doing. Using this, the user should be able
to determine when enough labels have been assigned. The app provides the
user with an option to **auto-label** the rest of the corpus. The app also
allows the user to **export a trained machine-learning classifier** (.pkl) that 
can be used outside of the app to assign labels to similar texts.

## Prerequisites
The web app can be used in one of 3 ways.

### 1. Using PIP Requirements
```
$ pip install -r requirements.txt
```

```
$ git clone https://github.com/Tak-Man/ML-rapid-text-labeling-app.git
```

```
$ python main.py
```

### 2. Create a Conda environment
This was the project development teams preferred method for setting up the environment required to run this code.
```
$ conda env create -f environment.yml
```

```
$ git clone https://github.com/Tak-Man/ML-rapid-text-labeling-app.git
```

```
$ python main.py
```

### 3. Accessing the Web App URL
The web app is available here: [ML-rapid-text-labeling-app](http://ml-rapid-text-labeling-app.herokuapp.com/)

As at 2021/12/12 a live version of the web app is available at this url. 
Please note that there is no guarantee that the web app will always be available here. 
Also the app is for demonstration purposes and can accommodate single-user use only.


## Contributors
* [https://github.com/Tak-Man](https://github.com/Tak-Man)
* [https://github.com/michp-ai](https://github.com/michp-ai)
