# Mozilla-Automatic-Bug-Triaging
Predict the software component to which the bug should belong to, given the past data about bugs and the components to which they were associated.

### Problem:
Bugzilla is a bug tracking system from Mozilla. It is used to develop and maintain hundreds of software products, consisting of thousands of components. As new bugs are submitted into the system, their triaging (categorization by products and software components) becomes time and resource consuming, as many bug reporters are not closely familiar with internal division of software they use into products and component. Thus, members of Mozilla development team are assigned to be gatekeepers, who categorize and prioritize incoming bugs. This manual bug triaging consumes time of experienced engineers, which may be spent better, otherwise.

In this challenge we were given real data from the bug tracking system, and asked to implement an algorithm for automatic bug triaging. Given details of a submitted bug, the algorithm should deduce the software component that bug belongs to, and the confidence score of this deduction. Find more about the problem [here](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17280&pm=15089).

### Approach:
* Straightforward multiclass classification using Deep Learning.
* Used multiple pre-trained embeddings and slightly different networks.
* Did 10-fold stratified cross validation on each model, a model’s predictions are itself mean of predictions of all 10 folds.
* Took weighted average of all the model predictions.

### Neural network layers:
* Bidirectional GRU layers.
* Attention layer.
* Dense layers.
* Dropouts.
 All the models were combination of these layers.

### Pre-trained embeddings:
* Fasttext common crawl.
* Glove common crawl.
* Fasttext common crawl on subwords.
* Fasttext wiki pretrained.

### Input to neural networks:
* I used only the bug description (short+long) as input to the models.
* Tried using other features too but it worsened the results.
* Tried giving weights to the classes to avoid class imbalance but it also worsened the results
