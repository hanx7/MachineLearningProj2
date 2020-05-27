### reading doc2vec_X file
print('reading files')
GLOBAL_learning_rate = 0.001
GLOBAL_hidden_units = 128
GLOBAL_layers = 3
GLOBAL_batch_size = 128
GLOBAL_diff_threshold = 0.0001
GLOBAL_forget_bias = 1.0
GLOBAL_vec_num = 50

fp_train = open('review_text_train_doc2vec200.csv', 'r')
fp_test = open('review_text_test_doc2vec200.csv', 'r')

all_sentences = []
for line in fp_train.readlines():
    vectors = [s.strip() for s in line.split(',')]
    all_sentences.append(vectors)
fp_train.close()

all_test_sentences = []
for line in fp_test.readlines():
    vectors = line.split(',')
    all_test_sentences.append(vectors)
fp_test.close()

fp_train_tags = open('review_meta_train.csv', 'r')
all_tags = []
all_votes = []
first_line = True
for line in fp_train_tags.readlines():
    if not first_line:
        votes_tag = line.strip().split(',')[4:]
        votes = votes_tag[0:3]
        tag = votes_tag[-1]
        all_votes.append(votes)
        all_tags.append(tag)
    else:
        first_line = False
fp_train_tags.close()

test_votes = []
fp_test_tags = open('review_meta_test.csv', 'r')
first_line = True
for line in fp_test_tags.readlines():
    if not first_line:
        line_votes = line.strip().split(',')[4:]
        test_votes.append(line_votes)
    else:
        first_line = False
fp_test_tags.close()

for i in range(0, len(all_votes)):
    all_sentences[i] = all_sentences[i] + all_votes[i]

for i in range(0, len(test_votes)):
    all_test_sentences[i] = all_test_sentences[i] + test_votes[i]

all_tags_vec = []
for tag in all_tags:
    if tag == '1':
        all_tags_vec.append([1, 0, 0])
    elif tag == '3':
        all_tags_vec.append([0, 1, 0])
    elif tag == '5':
        all_tags_vec.append([0, 0, 1])


all_sentences_embeddings = []
for line in all_sentences:
    line_float = []
    for item in line:
        line_float.append(float(item))
    all_sentences_embeddings.append(line_float)


all_test_sentences_embeddings = []
for line in all_test_sentences:
    line_float = []
    for item in line:
        line_float.append(float(item))
    all_test_sentences_embeddings.append(line_float)

### Normalization
from sklearn.preprocessing import normalize as nm

print('normalization')
all_embeddings = list(nm(all_sentences_embeddings + all_test_sentences_embeddings))
all_sentences_embeddings = all_embeddings[: len(all_sentences_embeddings)]
all_test_sentences_embeddings = all_embeddings[len(all_sentences_embeddings): ]

all_tags_num = []
for item in all_tags:
    all_tags_num.append(int(item))
print(all_tags_num)

print('file pre-processing finished ...')


### GNB
### GNB
### GNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# gnb = GaussianNB()
# mnb = MultinomialNB()
# y_pred = gnb.fit(all_sentences_embeddings, all_tags_num).predict(all_test_sentences_embeddings)
# print(y_pred)
#
# y_pred = list(y_pred)
# fp_pred = open('../data/res/predictions_gnb.txt', 'w+')
# res_line = 'Instance_id,rating\n'
# for i in range(0, len(y_pred)):
#     res_line += str(i + 1) + ',' + str(y_pred[i]) + '\n'
# fp_pred.write(res_line)
# fp_pred.close()

### SVM
### SVM
### SVM
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

### produce test results
X_train = np.array(all_sentences_embeddings)
y_train = np.array(all_tags_num)

### training splits
# X_train, X_test, y_train, y_test = train_test_split(all_sentences_embeddings, all_tags_num, test_size=0.05, random_state=0)


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model = clf.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print('SVM = ')
# print((y_test == y_pred).sum() / len(X_test))
# print()

print('write to file')
y_pred = list(clf.predict(all_test_sentences_embeddings))

fp_pred = open('predictions_svm.txt', 'w+')
# fp_pred = open('../data/res/predictions_linear_svc.txt', 'w+')
res_line = 'Instance_id,rating\n'
for i in range(0, len(y_pred)):
    res_line += str(i + 1) + ',' + str(y_pred[i]) + '\n'
fp_pred.write(res_line)
fp_pred.close()


### KNN
### KNN
### KNN

# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=15)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print((y_test == y_pred).sum() / len(X_test))

### LogisticRegress

from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# print('LogicReg')
# print((y_test == y_pred).sum() / len(X_test))
# print()

# print('write to file')
# y_pred = list(clf.predict(all_test_sentences_embeddings))
#
# fp_pred = open('../data/res/predictions_logicReg.txt', 'w+')
# res_line = 'Instance_id,rating\n'
# for i in range(0, len(y_pred)):
#     res_line += str(i + 1) + ',' + str(y_pred[i]) + '\n'
# fp_pred.write(res_line)
# fp_pred.close()

### Perception
### Perception
### Perception

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures

# clf = Perceptron(fit_intercept=False, max_iter=100, tol=None, shuffle=True).fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print('LogicReg')
# print((y_test == y_pred).sum() / len(X_test))
# print()



