# Amazon-Product-Reviews

Text Classification Problem :
Project Explores different classification algorithms on a corpus of Amazon Product Reviews.
Original data set was of 3 million instances(with attributes TITLE, RATING, REVIEWS), so the
data had to be cut short using https://www.splitcsv.com/ .

Preprocessing : 
Some attributes were discarded (namely TITLE) though it could be appended to REVIEWS too and new
column LIKED was added based on the RATING attribute. Value of LIKED, for a review was equal to 1 if rating was 
more than 3 else it was 0. The was a preprocessed furthur by stemming,making all words lowercase and retaining 
only the text and discarding any symbols (!, ?, etc) .

Vectorisation/Bag of Words :
We used TfIdfVectoriser to create the BOW model instead of CountVectoriser
CountVectorizer just counts the word frequencies. With the TFIDFVectorizer the value 
increases proportionally to count, but is offset by the frequency of the word in the corpus. -

Modelling :
We created 4 models for  classifying namely Logistic Regression, GaussianNB, MultnomialNB and DecisionTree.
Logistic Regression gave the best reslts with 78% accuracy and and F1 score of 81%.where as Decision Tree Classifier
performed the worst.

![alt text](https://raw.githubusercontent.com/abhinavthapper31/Amazom-Product-Reviews/path/to/Figure_1.png)

Conclusion : Try other methods like SVC and XGBoost to check if accuracy increases. Also try other ways of vectorising
Word2Vec, Doc2Vec etc..




