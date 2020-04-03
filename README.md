# Mod 4 Code Challenge: Product Reviews

This assessment is designed to test your understanding of these areas:

1. Data Engineering
    - Understanding an existing ETL pipeline
    - Feature scaling
2. Deep Learning with Neural Networks
    - Creating a TensorFlow neural network model
    - Fitting the model on training data
    - Hyperparameter tuning
    - Model evaluation on test data
3. Business Understanding and Technical Communication
    - Advising a business on what kind of model architecture to use

**Unlike previous challenges, we have provided you with a notebook containing some pre-existing code.**  You may continue working in that same notebook, or start a new notebook and copy over the relevant pieces of code before starting with the new task.

Make sure that your code is clean and readable, and that each step of your process is documented. For this challenge each step builds upon the step before it. If you are having issues finishing one of the steps completely, move on to the next step to attempt every section.  There will be occasional hints to help you move on to the next step if you get stuck, but attempt to follow the requirements whenever possible.

### Business Understanding

Northwind Trading Company allows customers to leave reviews, but those reviews do not have customer-facing "star ratings".  Instead, customers are free to write text, and other customers can vote on whether the review was helpful.  They find that this is a good trade-off between helping customers make informed decisions about products, and avoiding having any products go unsold because of poor ratings.

Internally, Northwind is interested to know which of these reviews are positive, and which are negative.  **A previous employee of the company has already built a Random Forest Classifier model to perform this classification task.**

Northwind management has heard great things about using Artificial Intelligence for this kind of task, especially Neural Networks like TensorFlow.  **You have been instructed to build a TensorFlow model and advise the company on whether they should switch from the Random Forest Classifier to the TensorFlow model.**

In either case, you want a **classification model** that optimizes for **accuracy**.

### Data Understanding

The data has already been described, imported, and preprocessed in the notebook within this directory.

### Data Preparation

A train-test split has already been performed.

Additionally, there is already a pipeline in place that drops some columns and converts all text columns into a numeric format for modeling.

**Your only additional data preparation task is feature scaling.**  Tree-based models like Random Forest Classifiers do not require scaling, but TensorFlow neural networks do.

There are two main strategies you can take for this task:

#### Scaling within the existing pipeline

If you are comfortable with pipelines, this is the more polished/professional route.

1. Add a `StandardScaler` as the final step in the pipeline
2. Generate a new `X_train_transformed` by calling `.fit_transform` again on the pipeline
3. Generate a new `X_test_transformed` by calling `.transform` again on the pipeline

#### Scaling after the pipeline has finished

This is a better strategy if you are not as comfortable with pipelines.

1. Instantiate a `StandardScaler` object
2. Generate a new `X_train_transformed` by calling `.fit_transform` on the scaler object, after you have called `.fit_transform` on the pipeline
3. Generate a new `X_test_transformed` by calling `.transform` on the scaler object, after you have called `.transform` on the pipeline

If you are getting stuck at this step, skip it.  The model will still be able to fit, although the performance will be worse.  Keep in mind whether or not you scaled the data in your final analysis.

### Modeling

Build a neural network classifier.  Specifically, use the `keras` submodule of the `tensorflow` library to build a multi-layer perceptron model with the `Sequential` interface.

See the [`tf.keras` documentation](https://www.tensorflow.org/guide/keras/overview) for an overview on the use of `Sequential` models. See the [Keras layers documentation](https://keras.io/layers/core/) for descriptions of the `Dense` layer options.  

1. Instantiate a `Sequential` model
2. Add an input `Dense` layer.  You'll need to specify a `input_shape` = (11275,) because this is the number of features of the transformed dataset.
3. Add one or more `Dense` hidden layers.  They can have any number of units, but keep in mind that more units will require more processing power.  We recommend an initial `units` of 64 for processing power reasons.
4. Add a final `Dense` output layer.  This layer must have exactly 1 unit because we are doing a binary prediction task.
5. Compile the `Sequential` model
6. Fit the `Sequential` model on the preprocessed training data (`X_train_transformed`).  We recommend an initial `batch_size` of 50 and `epochs` of 5 for processing power reasons.

### Model Tuning + Feature Engineering

If you are running out of time, skip this step.

Tune the neural network model to improve performance.  This could include steps such as increasing the units, changing the activation functions, or adding regularization.

We recommend using using a `validation_split` of 0.1 to understand model performance without utilizing the test holdout set.

You can also return to the preprocessing phase, and add additional features to the model.

### Model Evaluation

Choose a final `Sequential` model, add layers, and compile.  Fit the model on the preprocessed training data (`X_train_transformed`, `y_train`) and evaluate on the preprocessed testing data (`X_test_transformed`, `y_test`) using `accuracy_score`.

### Technical Communication

Write a paragraph explaining whether Northwind Trading Company should switch to using your new neural network model, or continue to use the Random Forest Classifier.  Beyond a simple comparison of performance, try to take into consideration additional considerations such as:

 - Computational complexity/resource use
 - Anticipated performance on future datasets (how might the data change over time?)
 - Types of mistakes made by the two kinds of models

You can make guesses or inferences about these considerations.

**Include at least one visualization** comparing the two types of models.  Possible points of comparison could include ROC curves, colorized confusion matrices, or time needed to train.
