# LLM---Detect-AI-Generated-Text

In recent years, large language models (LLMs) have become increasingly sophisticated, capable of generating text that is difficult to distinguish from human-written text. we are tasked with finding whether the text is AI written or not. if so we have to find the percentage.


We will implement this Using the concept of Naive Bayes classifier(NBC).


What is NBC:


NBC is a supervised machine learning algorithm used for classification tasks, such as text classification.


It is based on Bayes' theorem, which assumes that all features in the input data are independent of each other.


There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable.


Naive Bayes classifiers can be used to identify faces, predict weather, diagnose patients, and indicate if a patient is at high risk for certain diseases and conditions.


Implementation:

Import Libraries and Download NLTK Resources:




This part is focused on downloading necessary resources from the NLTK (Natural Language Toolkit) library. NLTK is a popular Python library used for working with human language data (text). It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.


nltk.download('punkt'):

Purpose: This command downloads the 'punkt' tokenizer models. The 'punkt' package is a pre-trained tokenization model used to divide a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It's especially useful for tokenizing sentences in different languages.


Usage in Project: It's used for sentence segmentation or tokenization in your text processing tasks.



nltk.download('stopwords'):

Purpose: This command downloads a set of 'stopwords'. Stopwords are words that do not contribute much meaning to a sentence. They are usually removed from texts during processing to reduce the amount of data to be analyzed. Examples of stopwords in English are "is", "and", "the", etc.


Usage in Project: In your project, this  used to filter out these stopwords from your text data, which is a common preprocessing step in various NLP tasks such as sentiment analysis, topic modeling, or text classification.


Define Text Preprocessing Functions : 

The function preprocess_text is designed to preprocess a given text for natural language processing tasks. This function performs several standard preprocessing steps which are crucial for text analysis. Let's break down what each line of code does




preprocess_text:

Purpose: To preprocess a given text string for subsequent natural language processing tasks.

Input: A string text that represents the text to be processed.

text = text.lower()

Lowercasing: Converts all characters in the text to lowercase. This is a 											common practice in text preprocessing to ensure that the algorithm treats words like "The" and "the" as the same word.

text = re.sub(r'\s+', ' ', text):

		Removing Extra Spaces: Uses regular expression to replace one or more 

		whitespace characters (\s+) with a single space (' ').This step cleans the text 

		by removing unnecessary spaces, tabs, or newlines.

tokens = word_tokenize(text):

			Tokenization: Breaks the text into individual words or tokens.

			Tokenization is a fundamental step in text preprocessing, turning a string 

			or a document into tokens (smaller chunks), making it easier to assign 

			meaning to the text or analyze its structure.

tokens = [word for word in tokens if word.isalpha()]:

			Removing Non-Alphabetic Tokens: Filters out any tokens that are not 

			purely alphabetic. This step removes numbers and punctuation, focusing 

			the analysis on words only.

stop_words = set(stopwords.words('english')):

			Stopwords List: Creates a set of stopwords using NLTK's built-in English 

			stopwords list. Stopwords are common words that usually do not carry 

			significant meaning and are often removed in NLP task

tokens = [word for word in tokens if word not in stop_words]:

			Removing Stopwords: Filters out stopwords from the tokens. This step

			ensures that the focus is on the more meaningful words in the text.


Output: Returns tokens, which is a list of processed words from the original text.


Usage: This function is useful in a variety of NLP applications such as sentiment analysis, topic modeling, and text classification. By preprocessing the text, it removes irrelevant or redundant information, helping in focusing on the important features of the text.



Define Vocabulary Building Function : 

he build_vocabulary function  provided is designed to create a vocabulary from a dataset of tokenized texts. This vocabulary is a dictionary mapping words to their unique indices, but only includes words that appear at least 5 times in the entire dataset. Let's break down the function to understand its components




build_vocabulary:


Purpose: To build a vocabulary dictionary where each unique word that appears at least 		

		5 times in the dataset is assigned a unique index.


Input: data, which is expected to be an iterable (like a list) of tokenized texts (where 

	   each text is itself an iterable of words/tokens).


Process and Explanation:

	1. Initialize Vocabulary Dictionary:

vocabulary = defaultdict(int): 

		●      Initializes vocabulary as a defaultdict of type int. In this defaultdict, 

			every new key will have a default value of 0. This is particularly useful for 

			counting occurrences of words.

	2. Count Word Occurrences:

		●      The nested for loops iterate over each text in data, and then over each 

			word in text.

		●      for text in data iterates through each tokenized text in the dataset.

		●      for word in text iterates through each word/token in the tokenized text.

		●      vocabulary[word] += 1 increments the count for each word. This line 

			counts the number of times each word appears in the dataset.

	3. Filter and Assign Indices:

		●      The return statement {word: index for index, word in 

			enumerate(vocabulary) if vocabulary[word] >= 5} constructs a dictionary 

			where each word is a key, and its value is a unique index (generated 

			using enumerate). However, it only includes words that have appeared 5 

			or more times in the entire dataset (if vocabulary[word] >= 5).


Output: A dictionary where keys are words and values are their respective indices, filtered to include only words that appear at least 5 times in the data.


Usage :

	●      Vocabulary Building for NLP Models: This function is crucial in NLP tasks 

		like text classification or language modeling, where you often need to 

		convert words to numerical indices for machine learning models.

	●     Reducing Dimensionality: By only including words that appear at least 5 times, 

		the function helps in reducing the dimensionality of the data, focusing on 

		more frequent and potentially more relevant words.

	●     Preprocessing for Embedding Layers: In deep learning models, especially 

		those using embedding layers, such a vocabulary is necessary to map words 

		to embeddings.



Define Naive Bayes Classifier Training Function : 


The function train_naive_bayes is designed to train a Naive Bayes classifier for text classification. Naive Bayes classifiers are popular in Natural Language Processing (NLP) for tasks such as sentiment analysis, spam detection, and topic classification. Let's dissect this function to understand its workings




Function: train_naive_bayes:


Purpose: To train a Naive Bayes classifier using a given dataset and a predefined vocabulary.


Input:

●      data: A DataFrame expected to contain the training data, with features and 

		  labels.

●      vocab: A vocabulary dictionary mapping words to their indices.


Process and Explanation:


Initialize      Word Counts and Class Counts:

	●  word_counts = {class_: defaultdict(int) for class_ in np.unique(data['generated'])}: 

		Initializes a dictionary to count the occurrences of each word in each class. 

		The keys are the unique class labels from the 'generated' column of data.

	●  class_counts = defaultdict(int): 

		A dictionary to count the number of documents in each class.

     2. Count Words and Classes:

	●      The function iterates over each row in the data DataFrame.

	●      label = row['generated']: Extracts the class label for the current document.

	●      class_counts[label] += 1: Increments the count for the current class.

	●      It then iterates through each word in the 'processed_text' field of the row. If 

		the word is in the provided vocabulary, it increments the count of that word 

		in the context of the current class.


      3. Calculate Probabilities:

	●      total_docs = len(data): 

			Counts the total number of documents in the dataset.

	●      word_probs: A nested dictionary where each outer key is a class, and the 

		inner dictionary maps each word in the vocabulary to its conditional 

		probability given the class.

	●      The word probabilities are calculated using the formula: 

		(word count in class+1)/(total words in class+length of vocabulary). This 

		implements Laplace smoothing.

	●      class_probs: A dictionary mapping each class to its probability, calculated as 

		the number of documents in the class divided by the total number of 

		documents


Output: Two dictionaries, class_probs and word_probs, representing the probabilities of classes and the conditional probabilities of words within those classes, respectively


Usage:

	●      Text Classification: This trained model can be used to classify new text 

		 documents into one of the classes.

	●      Naive Bayes Algorithm: It's based on the Naive Bayes algorithm, which 

		 assumes independence between features (words in this case). Despite this 

		 simplification, Naive Bayes classifiers often perform remarkably well on text 

		 data.

	●      Foundational Model in NLP: This kind of model is foundational in NLP and 

		 serves as a baseline for many text classification tasks.


 Define Classification Function : 

The classify function  is designed to classify a given text into one of the predefined classes using a trained Naive Bayes model. It takes into account the class probabilities, word probabilities within each class, and a vocabulary that was previously built. Let's break down this function to understand how it works:




Function: classify


Purpose: To classify a given text into one of the classes based on a trained Naive Bayes model.


Input:

●      text: The text to be classified.

●      class_probs: A dictionary of probabilities for each class.

●      word_probs: A nested dictionary of word probabilities for each word in each class.

●      vocab: A dictionary representing the vocabulary.


Process and Explanation:

Preprocess the Text:

		● text_words = set(preprocess_text(text)): 

		Preprocesses the input text (lowercasing, removing non-alphabetic characters, 

		tokenizing, and removing stopwords) and converts it to a set of words. The 

		preprocessing function used is assumed to be the one you provided earlier.

	2. Initialize      Class Scores:

		● class_scores = {class_: np.log(class_prob) for class_, class_prob in 

					    class_probs.items()}: 

		Initializes a dictionary to store the scores for each class. The score is initiated 

		with the logarithm of the class probability. Using logarithms helps in avoiding 

		underflow issues common in probabilistic computations.

	3. Calculate Scores for Each Class:

		● The function iterates through each class and their respective word 

		   probabilities.

		● For each word in the preprocessed text_words, if the word is in the 

		   vocabulary, it updates the score of the class by adding the logarithm of the 

		   word's probability in that class.

		● If a word is not found in word_probs, it uses a smoothing technique 

		   (Laplace smoothing) to handle unknown words: 1 / (len(vocab) + 

		   sum(word_prob.values())). This prevents multiplication by zero probability.

	4. Classify the Text:

		● return max(class_scores, key=class_scores.get): The function returns the 

			class with the highest score. This is the essence of the Naive Bayes 

			classification - choosing the class that maximizes the posterior 

			probability.


Usage :

	● Text Classification: This function is used to classify texts into categories based on 

		the learned probabilities from the Naive Bayes model.

	● Naive Bayes in Action: It showcases how a probabilistic model like Naive Bayes 

		can be applied to real-world text data.

	● Handling Unseen Data: The function includes handling of words not seen in the 

		training data (out-of-vocabulary words), which is a common scenario in text 

		classification tasks.


Train Naive Bayes Classifier and Evaluate on Development Data : 


how to train a Naive Bayes classifier and then evaluate its performance on a development dataset. Let's break down the steps:




Training the Naive Bayes Classifier:

class_probs, word_probs = train_naive_bayes(train_data, vocab):

	● This line calls the train_naive_bayes function with the training data (train_data) 

		and a vocabulary (vocab).

	● The function returns two items:

	● class_probs: A dictionary of probabilities for each class.

	● word_probs: A nested dictionary containing the probabilities of each word 

		within each class.

	● These outputs are used to represent the trained Naive Bayes model.


Evaluating the Classifier

	1. Applying the Classifier to Development Data:

	    dev_data['predicted'] = dev_data['text'].apply(lambda text: classify(text, 

	    class_probs, word_probs, vocab)):

		● This line applies the classify function to each text in the development 

			dataset (dev_data).

		● The classify function uses the trained model to predict the class for each

			text.

		● The predictions are stored in a new column predicted in the dev_data

			DataFrame. 

	2. Calculating Accuracy: 

		accuracy = np.mean(dev_data['predicted'] == dev_data['generated']):

		● This line calculates the accuracy of the classifier.

		● It compares the predicted classes (dev_data['predicted']) with the actual 

			classes (dev_data['generated']).

		● The expression dev_data['predicted'] == dev_data['generated'] creates a 

			boolean series where True represents a correct prediction and False an 

			incorrect one.

		● np.mean() computes the average of this series, effectively calculating the 

			proportion of correct predictions, which is the accuracy of the model.


	3. Printing the Accuracy: print(f'Accuracy on development set: {accuracy*100}%'):

		● This prints out the accuracy as a percentage.


Usage : 

	● Model Validation: This process is crucial for understanding how well the Naive 

		Bayes model performs on unseen data. It's a common practice in machine 

		learning to evaluate models on a separate dataset (development set) to check 

		their generalization capabilities.

	● Performance Metric: Accuracy is a straightforward metric that tells you the 

		proportion of correctly classified instances. It's particularly useful when the 

		classes are balanced.

	● Iterative Improvement: Based on the accuracy, you might decide to adjust the 

		preprocessing, tweak the model, or try different features to improve 

		performance. 



Co	mpare effect of smoothing : 

This updated train_naive_bayes function now includes the capability to use different smoothing techniques for handling zero probabilities in the Naive Bayes model. Let's break down the enhancements and how they affect the training process.




Function: train_naive_bayes

Purpose: To train a Naive Bayes classifier with the flexibility to use different smoothing 

		techniques.

Input:

	● data: A Data Frame expected to contain the training data, with features and 

		labels.

	● vocab: A vocabulary dictionary mapping words to their indices.

	● smoothing_method (optional): A string indicating the smoothing technique to 

		use ('laplace' or 'add-k').

	● smoothing_parameter (optional): A numerical value representing the smoothing 

		parameter (commonly denoted as alpha or k).


Enhanced Process and Explanation:

	1. Word Counts and Class Counts Initialization:

		● The initialization of word_counts and class_counts remains the same, 

			counting occurrences of words per class and document frequencies of 

			each class.


	2. Word and Class Probabilities Calculation:

		● The calculation of class_probs remains the same.

		● word_probs is initialized as a nested dictionary for each class.

		● The function now iterates through each class and word to calculate word 

			probabilities, considering the specified smoothing method.


	3. Smoothing Techniques:

		Laplace Smoothing ('laplace'):

		● word_probs[class_][word] = (word_count.get(word, 0) + 

			smoothing_parameter) / (class_counts[class_] + smoothing_parameter * 

			len(vocab))

		● This adds the smoothing_parameter to the word count and a multiple of the 

			smoothing parameter to the denominator, proportional to the vocabulary 

			size. 

		● It's a common technique to handle zero probabilities, ensuring that every 

			word has a non-zero probability.

	        Add-k Smoothing ('add-k'):

		● word_probs[class_][word] = (word_count.get(word, 0) + 

			smoothing_parameter) / (class_counts[class_] + smoothing_parameter)

		● Similar to Laplace smoothing, but the denominator is incremented by a 

			constant smoothing_parameter, not dependent on the vocabulary size.

		● Useful when the dataset is large, and the effect of the vocabulary size on 

			the denominator needs to be controlled.


Error Handling for Invalid Smoothing Methods:

	● Raises a ValueError if an invalid smoothing method is specified.


Usage :

	● Flexibility in Smoothing Techniques: Different smoothing methods can be more 

		effective depending on the dataset's characteristics. This function provides 

		the flexibility to experiment with different methods.

	● Robustness to Zero Frequencies: Smoothing helps in handling cases where a 

		word present in the test set was not observed in the training set for a 

		particular class.

	● Enhanced Model Performance: Proper smoothing can significantly improve the 

		performance of a Naive Bayes classifier, especially on small or sparse 

		datasets.	


Accuracy on Development Set : 





Predicting the Top 10 words of both Human and LLM essay's:





Final Accuracy after smoothing Effect:



My Contribution:

I have done my contribution while building a vocabulary and also i had done my contribution by using two smoothing techinques they are Lapace and add-k smoothing


What is Laplace Smoothing:

Laplace smoothing, also known as additive smoothing or Lidstone smoothing, is a technique used to smooth categorical data.

Laplace smoothing helps tackle the problem of zero probability in the Nave Bayes machine learning algorithm by pushing the likelihood towards a value of 0.5, i.e., the probability of a word equal to 0.5 for both the positive and negative reviews.


What is Add-k Smoothing:
Add-k smoothing, also known as Laplace smoothing or add-one smoothing, is a technique used in data mining and natural language processing to address the problem of zero probabilities in probability estimation. It is commonly applied when dealing with categorical data, such as in text classification or language modeling.


The basic idea behind add-k smoothing is to add a small constant, often denoted as "k," to the count of each possible outcome for a given feature, regardless of whether that outcome is observed in the training data. This ensures that no probability estimate is zero, even for events that did not occur in the training set.



Source Code: https://www.kaggle.com/ruthwikgangavelli/llm-detect-ai-generated-text 


References :
https://www.kaggle.com/competitions/llm-detect-ai-generated-text 

https://www.ibm.com/topics/naive-bayes 

https://en.wikipedia.org/wiki/Naive_Bayes_classifier

https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/ 

https://www.simplilearn.com/tutorials/machine-learning-tutorial/naive-bayes-classifier 

https://en.wikipedia.org/wiki/Additive_smoothing 

https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece 

https://chat.openai.com/ 
