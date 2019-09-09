# DC-DestinationPrediction
> Solution of Intelligent Prediction Competition of Automobile Destination in DC Competition  

**[Auto Destination Intelligent Prediction Competition website](http://www.dcjingsai.com/common/cmpt/%E6%B1%BD%E8%BD%A6%E7%9B%AE%E7%9A%84%E5%9C%B0%E6%99%BA%E8%83%BD%E9%A2%84%E6%B5%8B%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)**  

## My ranking
![img](https://github.com/lcylmhlcy/DC-DestinationPrediction/raw/master/img/dc.png)  
  
**refer to [website](http://www.dcjingsai.com/common/cmpt/%E6%B1%BD%E8%BD%A6%E7%9B%AE%E7%9A%84%E5%9C%B0%E6%99%BA%E8%83%BD%E9%A2%84%E6%B5%8B%E5%A4%A7%E8%B5%9B_%E6%8E%92%E8%A1%8C%E6%A6%9C.html)**

## Getting started
- Python 3.6 (Anaconda3)
- sklearn
- numpy
- pandas

## Introdution
1. **Task:** By learning the historic journey of some vehicles and training the model, the destination of the vehicle in this journey can be predicted with given vehicle id, time and departure place.  
2. **Dataset:** 
- Train_new.csv training set data, a total of 1495 814 samples, for January 1, 2018 to August 1, 2018 (including the data in the original test set).   
![img](https://github.com/lcylmhlcy/DC-DestinationPrediction/raw/master/img/1.png)
- Test_new.csv test set data, for the data extracted from September to October of the same year, except for no end_time, end_lat, end_lon fields, other fields with the training set data. A total of 58097 samples.

## Method
1. Computational Density Clustering (DBSCAN)
2. Generating test set block ID
3. Generate is_holiday and hot fields for training and test sets
4. Calculating conditional probability of Naive Bayesian algorithm based on training set
5. Calculate the prior probability according to the training set
6. Calculate the posterior probability
7. Generate latitude and longitude of each cluster label
8. write to CSV

The generated intermediate CSV files are in the folder **bayes(mkdir)**, as follows:  
![img](https://github.com/lcylmhlcy/DC-DestinationPrediction/raw/master/img/2.png)
