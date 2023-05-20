
# Fake-News-Detector

## Description
Fake-News-Detector is a machine learning project that utilizes the TfidfVectorizer and PassiveAggressiveClassifier algorithms to classify news articles as either fake or real. The project includes a web interface built with Flask and HTML, allowing users to input news articles and receive the classification results.

The project leverages the TF-IDF (Term Frequency-Inverse Document Frequency) approach to transform sentences into a matrix of numerical features. These features capture the importance of words in the articles based on their frequency within the document and across the corpus. The TfidfVectorizer implementation from the scikit-learn library is used for this purpose.

The transformed TF-IDF features are then used to train a model called PassiveAggressiveClassifier . This classifier is a type of online learning algorithm that is particularly effective for binary classification tasks, such as distinguishing between fake and real news.


## Usage

To use the Fake-News-Detector, follow these steps:

1. Clone the repository to your local machine:
```bash
  git clone https://github.com/Taha533/Fake-News-Detector.git
```
2. Install the required dependencies. You can use pip to install the necessary packages listed in the requirements.txt file:

```bash
  pip install -r requirements.txt
```

3. Run the Flask web application:

```bash
  python app.py
```
4. Access the web interface by opening your web browser.

5. In the input field, enter the news article you want to classify as fake or real.

6. Click the "Predict" button to obtain the classification result.

## Project Structure
The project repository is structured as follows:

**fake_news_detection.py**: This file contains the Flask application code responsible for handling web requests and serving the classification results.\

**classifier.pkl**: This file stores the trained model object in serialized form. The model is used for classifying news articles. 

**templates/**: This directory contains the HTML templates used by Flask to render the web interface.

**index.html**: The main HTML template for the web interface.

**static/**: This directory holds static files, such as CSS stylesheets used by the HTML templates.

**model_building.ipynb**: A Jupyter Notebook containing the code for training the model and generating the serialized model file.

## Model Training
The model_building.ipynb Jupyter Notebook provides an in-depth overview of the model training process. It covers the following steps:

- Loading and preprocessing the dataset.
- Transforming the sentences into a matrix of TF-IDF features using the TfidfVectorizer.
- Splitting the data into training and testing sets.
- Training a PassiveAggressiveClassifier using the TF-IDF features.
- Evaluating the model's performance using accuracy and confusion matrix.
- Saving the trained model in the serialized format for later use

## Dependencies
The project relies on the following dependencies, which can be installed using the provided requirements.txt file:

flask\
scikit-learn\
numpy\
pandas

You can install these dependencies by running the following command:

```bash
  pip install -r requirements.txt
```
## Screenshots:
![Home Page](https://github.com/Taha533/Fake-News-Detector/blob/main/HomePage.PNG?raw=true)



## License
This project is licensed under the [MIT License](https://github.com/Taha533/Fake-News-Detector/blob/main/LICENSE).

## Acknowledgments
This project was inspired by the need to tackle the issue of fake news and promote information credibility. The implementation is based on concepts and techniques from the field of natural language processing and machine learning. The scikit-learn library and Flask framework were instrumental in building this project.

## Contributions
Contributions to this project are welcome. If you would like.



    