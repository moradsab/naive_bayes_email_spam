import pandas
from pandas import read_csv
import re
import math


training_path="data_train.csv"
training_data=pandas.DataFrame(read_csv(training_path))
test_path="data_test.csv"
test_set=pandas.DataFrame(read_csv(test_path))


# Clean the training data we have

training_data['email']=training_data['email'].str.replace('\W',' ', regex=True)
training_data['email']=training_data['email'].str.replace('_',' ', regex=True)
training_data['email']=training_data['email'].str.lower()
training_data['email']=training_data['email'].str.split()
training_data.dropna()
test_set.dropna()

# Create vocabulary of words that occured in the training data

vocabulary=[]
for message in training_data['email']:
    if(type(message) == list):
        for word in message:
            vocabulary.append(word)
vocabulary=list(set(vocabulary))

# Count number of times the word occured in each email

word_occur_in_message = {unique_word: [0] * len(training_data['email']) for unique_word in vocabulary}
for i, message in enumerate(training_data['email']):
    if(type(message)==list):
        for word in message:
            word_occur_in_message[word][i] += 1
words_occurs_num = pandas.DataFrame(word_occur_in_message)
training_set = pandas.concat([training_data, words_occurs_num], axis=1)

# Split spam and non-spam messages

spam_messages=[]
non_spam_messages=[]

for row in training_data.itertuples():
    if(row[2]==1 and type(row[1])==list):
        spam_messages.append(row[1])
    elif(row[2]==0 and type(row[1])==list):
        non_spam_messages.append(row[1])

# Calculate P(Spam) , P(NonSpam) from all the data set

p_spam = len(spam_messages) / len(training_set)
p_non_spam = len(non_spam_messages) / len(training_set)


# calculate number of spam and non-spam words

spam_words_number=0
non_spam_word_number=0

for message in spam_messages:
    for word in message:
        spam_words_number+=1
        
for message in non_spam_messages:
    for word in message:
        non_spam_word_number+=1

# Calculate number of words in our vocabulary (features)

words_number = len(vocabulary)

# Laplace smoothing
alpha = 10

# Initiate parameters for counting number of occurs for each word in email

parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_non_spam = {unique_word:0 for unique_word in vocabulary}

for word in vocabulary:
    word_occurs_in_spam=0
    word_occurs_in_non_spam=0
    for message in spam_messages:
        word_occurs_in_spam+=message.count(word)
    p_spam_word=(word_occurs_in_spam+alpha)/(spam_words_number + alpha*words_number)
    parameters_spam[word]=p_spam_word
    
    for message in non_spam_messages:
        word_occurs_in_non_spam+=message.count(word)
    p_non_spam_word = (word_occurs_in_non_spam + alpha) / (non_spam_word_number + alpha*words_number)
    parameters_non_spam[word] = p_non_spam_word


# Now we have P(spam/non_spam),and the dataset number of occurs for each word in email
# Naive Bayes classifier depending on our dataset

def naive_bayes_classifier(message):

    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    p_spam_message = p_spam
    p_non_spam_message = p_non_spam

    for word in message:
        if word in parameters_spam:
            p_spam_message *= parameters_spam[word]

        if word in parameters_non_spam:
            p_non_spam_message *= parameters_non_spam[word]

    
    if p_non_spam_message >= p_spam_message:
        return int(0)
    elif p_spam_message > p_non_spam_message:
        return int(1)
    
# Run the classifier on our test set

test_set['predicted'] = test_set['email'].apply(naive_bayes_classifier)

# Count correct predections of the classifier

correct = 0
total = test_set.shape[0]

for email in test_set.itertuples():
    if(email[2]==email[3]):
        correct+=1

# Print the accuracy of the classifier

print(correct/total)