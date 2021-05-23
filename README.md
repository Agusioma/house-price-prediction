# house-price-prediction
## A House Price Prediction project
This is a project undertaken from the book **Hands-On Machine Learning with Scikit-Learn & TensorFlow CONCEPTS, TOOLS, AND TECHNIQUES TO BUILD INTELLIGENT SYSTEMS** by **Aurélien Géron**.

![book-cover](https://github.com/Agusioma/house-price-prediction/blob/main/book-cover.png)

It uses data from the *California Housing Prices dataset from the StatLib repository*
Based on Supervised Learning, it predicts the price of a house in Carlifonia using the number of rooms, median income, ocean procimity etc.

### Methods and Technologies used
Python language and its libraries such as Numpy, Pandas, Matplotlib and mainly Scikit-Learn were deployed.

I tested both *Random and Stratified Sampling* methods for splitting training and testing data but settled on the latter.

For handling text and categorical attributes from the dataset, I used the *LabelBinarizer* method provided by Scikit-Learn.

A transformation pipeline was fed using a custom transformer and Scikit-Learn's transformers for preparing the data to be fed to the models.

Three prediction methods were tested at first:
- Linear Regression
- Decision Trees
- Random Forest

I settled on the Random Forest method because it had the lowest Root Mean Squared Error.

*Grid Search Optimization* was used to optimize the model in which various hyperparameters were used.
 
  > The code will be improved time to time and OOP will be used. You can also uncomment relevant lines.
