from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn import tree
import json
# 1. delete all samples that are debatable: train/dev data: (2,3) and test data: 3
# 2. train samples: 200, dev/test samples: 50
# 3. for binary classifier: train/dev/test data, 
# input_feature: [[TF_IDF_cosine_similarity, word_to_vector_cosine_similarity],...]
# label: [1, 0, ...], paraphrases: 1, non-paraphrases: 0
def data_preprocessing(file_name):
    global data_path, train_samples, dev_samples, test_samples
    global train_input, train_label, dev_input, dev_label, test_input, test_label, test_sentence_pair_index_dict

    test_count = 0
    with open(f"{data_path}/{file_name}_no_debatable.data", "w+") as write_file, \
    open(f"{data_path}/{file_name}.data", "r") as read_file:
        for line in read_file:
            line_list = line.split("\t")
            sentence1 = line_list[2]
            sentence2 = line_list[3]
            label = line_list[4]
            TF_IDF_cosine = TF_IDF_cosine_similarity(sentence1, sentence2)[1]
            word_to_vector_cosine = word_to_vector_cosine_similarity(sentence1, sentence2)[1]

            if file_name in ['train', 'dev']:
                if label != "(2, 3)":
                    if file_name == 'train' and train_samples > 0:
                        write_file.write(line)
                        train_input.append([TF_IDF_cosine, word_to_vector_cosine])
                        train_label.append(0 if label in ["(1, 4)", "(0, 5)"] else 1)
                        train_samples -= 1
                        print(f"train_input: {train_input[-1]}, train_label: {train_label[-1]}")

                    elif file_name == 'dev' and dev_samples > 0:
                        write_file.write(line)
                        dev_input.append([TF_IDF_cosine, word_to_vector_cosine])
                        dev_label.append(0 if label in ["(1, 4)", "(0, 5)"] else 1)
                        dev_samples -= 1
                        print(f"dev_input: {dev_input[-1]}, dev_label: {dev_label[-1]}")

            elif file_name == 'test':
                if label != "3" and test_samples > 0:
                    test_sentence_pair_index_dict[test_count] = [sentence1, sentence2]
                    write_file.write(line)
                    test_input.append([TF_IDF_cosine, word_to_vector_cosine])
                    test_label.append(0 if label in ['0', '1', '2'] else 1)
                    test_samples -= 1
                    test_count += 1
                    print(f"test_input: {test_input[-1]}, test_label: {test_label[-1]}")

            sample_var_name = file_name + "_samples"
            if eval(sample_var_name) == 0:
                break



# baseline algorithm, characters match. Do alignment of the two sentences, and check the percentage of characters match.
# if the characters match is >= baseline_algo_threshold of all characters in the shortest sentence, paraphrases; otherwise, non-paraphrases
def baseline_algo(sentence1, sentence2):
    global baseline_algo_threshold
    match_number = 0
    short_sentence = sentence1 if len(sentence1) <= len(sentence2) else sentence2
    long_sentence = sentence2 if short_sentence == sentence1 else sentence1
    for each_char in range(len(short_sentence)):
        if short_sentence[each_char] == long_sentence[each_char]:
            match_number += 1
    match_percentage = match_number/len(short_sentence)
    print(f"baseline_algo: short_sentence: {short_sentence}, long_sentence: {long_sentence}\nmatch_number: {match_number}, short_sentence_length: {len(short_sentence)}, match_percentage: {match_percentage}")
    if match_percentage >= baseline_algo_threshold:
        return "yes"
    return "no"


# embed each sentence into a vector using TF_IDF (doesn't care semantics) and compute cosine similarity of the two vectors
def TF_IDF_cosine_similarity(sentence1, sentence2):
    global TF_IDF_cosine_similarity_threshold
    sentence_list = [sentence1, sentence2]
    vect = TfidfVectorizer(stop_words="english") 
    tfidf = vect.fit_transform(sentence_list)
    pairwise_similarity = tfidf * tfidf.T 
    cosine_similarity = pairwise_similarity.toarray()[0][1]
    print(f"TF_IDF_cosine_similarity: sentence1: {sentence1}, sentence2: {sentence2}\nvect: {vect}, tfidf: {tfidf.toarray()}\npairwise_similarity: {pairwise_similarity.toarray()}\ncosine_similarity: {cosine_similarity}")
    if cosine_similarity >= TF_IDF_cosine_similarity_threshold:
        return 'yes', cosine_similarity
    return 'no', cosine_similarity


# embed each sentence into a vector using a  pre-trained model (consider the semantics)"glove-twitter-25" and compute cosine similarity of the two vectors
def word_to_vector_cosine_similarity(sentence1, sentence2):
    word_emb_model = api.load("glove-twitter-25")
    print(f"word_to_vector_cosine_similarity: sentence1: {sentence1}, sentence2: {sentence2}")
    sentence1 = [token.lower() for token in sentence1.split() if token.lower() in word_emb_model.vocab]
    sentence2 = [token.lower() for token in sentence2.split() if token.lower() in word_emb_model.vocab]
    
    cosine_similarity = 0
    if (len(sentence1) > 0 and len(sentence2)>0):
        cosine_similarity = word_emb_model.n_similarity(sentence1,sentence2)
        print(f"After tokenization, sentence1: {sentence1}, sentence2: {sentence2}\ncosine_similarity: {cosine_similarity}")
        if cosine_similarity >= word_to_vector_cosine_similarity_threshold:
            return 'yes', cosine_similarity
    return 'no', cosine_similarity


def precision_recall(tp, fp, fn):
    precision = 0
    recall = 0
    if (tp + fp) != 0:
        precision = tp/(tp+fp)
    if (tp + fn) != 0:
        recall = tp/(tp+fn)

    return precision, recall


def macro_precision_recall(precision1, recall1, precision0, recall0):
    macro_precision = (precision1+precision0)/2
    macro_recall = (recall1+recall0)/2

    return macro_precision, macro_recall


def micro_precision_recall(tp1, tp0, fp1, fp0, fn1, fn0):
    micro_precision = (tp1 + tp0)/(tp1 + fp1 + tp0 + fp0)
    micro_recall = (tp1 + tp0)/(tp1 + fn1 + tp0 + fn0)

    return micro_precision, micro_recall


def f1_score(micro_precision, micro_recall, precision1, recall1, precision0, recall0):
    micro_f1_score = 0
    if micro_precision + micro_recall != 0:
        micro_f1_score = 2 * (micro_precision * micro_recall)/(micro_precision + micro_recall)

    f1_score1, f1_score0 = 0, 0
    if precision1 + recall1 != 0:
        f1_score1 = 2 * (precision1 * recall1)/(precision1 + recall1)
    if precision0 + recall0 != 0:
        f1_score0 = 2 * (precision0 * recall0)/(precision0 + recall0)
    macro_f1_score = (f1_score1 + f1_score0)/2

    return f1_score1, f1_score0, micro_f1_score, macro_f1_score


# feed dev_no_debatable.data into different kinds of algorithms and calculate the precision and recall
# evaluations: evaluate binary labels "0" and "1" separately, and calculate micro/macro precision/recall.
def feed_dev_test_into_algo():
    global algorithm_name, log_file, test_sentence_pair_index_dict
    dev_test = ["dev", "test"]
    for algo_index in range(len(algorithm_name)):
        log_test_sentence_file = open(f'{algorithm_name[algo_index]}_test.txt', 'w+')
        for data_name in dev_test:
            # tp1: true positive if to consider the label 1 as the positive class
            # tp0: true positive if to consider the label 0 as the positive class
            tp1, tp0, fp1, fp0, fn1, fn0 = 0, 0, 0, 0, 0, 0
            with open(f"{data_path}/{data_name}_no_debatable.data", "r") as read_file:
                for line in read_file:
                    line_list = line.split("\t")
                    sentence1 = line_list[2]
                    sentence2 = line_list[3]
                    if data_name == "dev":
                        ground_truth = "no" if line_list[4] in ["(1, 4)", "(0, 5)"] else "yes"
                    elif data_name == "test":
                        ground_truth = "no" if line_list[4] in ['0', '1', '2'] else "yes"
                    if algorithm_name[algo_index] == "baseline_algo":
                        prediction = eval(algorithm_name[algo_index])(sentence1, sentence2)
                    else:
                        prediction = eval(algorithm_name[algo_index])(sentence1, sentence2)[0]
                    if prediction == "yes" and ground_truth == "yes":
                        if data_name == 'test':
                            log_test_sentence_file.write(f"tp1: {sentence1}, {sentence2}\n")
                        tp1 += 1
                    elif prediction == "yes" and ground_truth == "no":
                        if data_name == 'test':
                            log_test_sentence_file.write(f"fp1/fn0: {sentence1}, {sentence2}\n")
                        fp1 += 1
                        fn0 += 1
                    elif prediction == "no" and ground_truth == "yes":
                        if data_name == 'test':
                            log_test_sentence_file.write(f"fn1/fp0: {sentence1}, {sentence2}\n")
                        fn1 += 1
                        fp0 += 1
                    else:
                        if data_name == 'test':
                            log_test_sentence_file.write(f"tp0: {sentence1}, {sentence2}\n")
                        tp0 += 1

                precision1, recall1 = precision_recall(tp1, fp1, fn1)
                precision0, recall0 = precision_recall(tp0, fp0, fn0)
                macro_precision, macro_recall = macro_precision_recall(precision1, recall1, precision0, recall0)
                micro_precision, micro_recall = micro_precision_recall(tp1, tp0, fp1, fp0, fn1, fn0)
                f1_score1, f1_score0, micro_f1_score, macro_f1_score = f1_score(micro_precision, micro_recall, precision1, recall1, precision0, recall0)
                log_file.write(f"\n{algorithm_name[algo_index]}, {data_name} data \
               \npostivie class: 1, tp1: {tp1}, fp1: {fp1}, fn1: {fn1}, precision: {precision1}, recall: {recall1}, f1_score1: {f1_score1}  \
               \npostive class: 0, tp0: {tp0}, fp0: {fp0}, fn0: {fn0}, precision: {precision0}, recall: {recall0}, f1_score0: {f1_score0} \
               \nmacro precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1_score: {macro_f1_score} \
               \nmicro preicsion: {micro_precision}, micro_recall: {micro_recall}, micro_f1_score: {micro_f1_score}\n")

        log_test_sentence_file.close()



def train_dev_test_knn():
    global test_sentence_pair_index_dict

    log_test_sentence_file = open('knn_test.txt', 'w+')
    # train the decision tree
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_input, train_label)

    data_name_list = ['dev', 'test']
    for data_name in data_name_list:
        # test on dev/test data
        data_input = eval(data_name + "_input")
        output = model.predict(data_input)
        data_label = eval(data_name + "_label")
        # tp1: true positive if to consider the label 1 as the positive class
        # tp0: true positive if to consider the label 0 as the positive class
        tp1, tp0, fp1, fp0, fn1, fn0 = 0, 0, 0, 0, 0, 0
        for predict_index in range(len(output)):
            if output[predict_index] == 1 and data_label[predict_index] == 1:
                if data_name == "test":
                    log_test_sentence_file.write(f"tp1: {test_sentence_pair_index_dict[predict_index]}\n")
                tp1 += 1
            if output[predict_index] == 1 and data_label[predict_index] == 0:
                if data_name == "test":
                    log_test_sentence_file.write(f"fp1/fn0: {test_sentence_pair_index_dict[predict_index]}\n")
                fp1 += 1
                fn0 += 1
            if output[predict_index] == 0 and data_label[predict_index] == 1:
                if data_name == "test":
                    log_test_sentence_file.write(f"fn1/fp0: {test_sentence_pair_index_dict[predict_index]}\n")
                fn1 += 1
                fp0 += 1
            if output[predict_index] == 0 and data_label[predict_index] == 0:
                if data_name == "test":
                    log_test_sentence_file.write(f"tp0: {test_sentence_pair_index_dict[predict_index]}\n")
                tp0 += 1

        precision1, recall1 = precision_recall(tp1, fp1, fn1)
        precision0, recall0 = precision_recall(tp0, fp0, fn0)
        macro_precision, macro_recall = macro_precision_recall(precision1, recall1, precision0, recall0)
        micro_precision, micro_recall = micro_precision_recall(tp1, tp0, fp1, fp0, fn1, fn0)
        f1_score1, f1_score0, micro_f1_score, macro_f1_score = f1_score(micro_precision, micro_recall, precision1, recall1, precision0, recall0)
        log_file.write(f"\nKNN, {data_name} data \
                       \npostivie class: 1, tp1: {tp1}, fp1: {fp1}, fn1: {fn1}, precision: {precision1}, recall: {recall1}, f1_score1: {f1_score1} \
                       \npostive class: 0, tp0: {tp0}, fp0: {fp0}, fn0: {fn0}, precision: {precision0}, recall: {recall0}, f1_score0: {f1_score0}\
                       \nmacro precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1_score: {macro_f1_score} \
                       \nmicro preicsion: {micro_precision}, micro_recall: {micro_recall}, micro_f1_score: {micro_f1_score}\n")
    log_test_sentence_file.close()




def train_dev_test_dt():
    global test_sentence_pair_index_dict

    log_test_sentence_file = open('dt.txt', 'w+')
    # train the decision tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_input, train_label)

    data_name_list = ['dev', 'test']
    for data_name in data_name_list:
        # test on dev/test data
        data_input = eval(data_name + "_input")
        output = model.predict(data_input)
        data_label = eval(data_name + "_label")
        # tp1: true positive if to consider the label 1 as the positive class
        # tp0: true positive if to consider the label 0 as the positive class
        tp1, tp0, fp1, fp0, fn1, fn0 = 0, 0, 0, 0, 0, 0
        for predict_index in range(len(output)):
            if output[predict_index] == 1 and data_label[predict_index] == 1:
                if data_name == "test":
                    log_test_sentence_file.write(f"tp1: {test_sentence_pair_index_dict[predict_index]}\n")
                tp1 += 1
            if output[predict_index] == 1 and data_label[predict_index] == 0:
                if data_name == "test":
                    log_test_sentence_file.write(f"fp1/fn0: {test_sentence_pair_index_dict[predict_index]}\n")
                fp1 += 1
                fn0 += 1
            if output[predict_index] == 0 and data_label[predict_index] == 1:
                if data_name == "test":
                    log_test_sentence_file.write(f"fn1/fp0: {test_sentence_pair_index_dict[predict_index]}\n")
                fn1 += 1
                fp0 += 1
            if output[predict_index] == 0 and data_label[predict_index] == 0:
                if data_name == "test":
                    log_test_sentence_file.write(f"tp0: {test_sentence_pair_index_dict[predict_index]}\n")
                tp0 += 1

        precision1, recall1 = precision_recall(tp1, fp1, fn1)
        precision0, recall0 = precision_recall(tp0, fp0, fn0)
        macro_precision, macro_recall = macro_precision_recall(precision1, recall1, precision0, recall0)
        micro_precision, micro_recall = micro_precision_recall(tp1, tp0, fp1, fp0, fn1, fn0)
        f1_score1, f1_score0, micro_f1_score, macro_f1_score = f1_score(micro_precision, micro_recall, precision1, recall1, precision0, recall0)
        log_file.write(f"\nDT, {data_name} data \
                       \npostivie class: 1, tp1: {tp1}, fp1: {fp1}, fn1: {fn1}, precision: {precision1}, recall: {recall1}, f1_score1: {f1_score1} \
                       \npostive class: 0, tp0: {tp0}, fp0: {fp0}, fn0: {fn0}, precision: {precision0}, recall: {recall0}, f1_score0: {f1_score0}\
                       \nmacro precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1_score: {macro_f1_score} \
                       \nmicro preicsion: {micro_precision}, micro_recall: {micro_recall}, micro_f1_score: {micro_f1_score}\n")
    log_test_sentence_file.close()


if __name__ == "__main__":
    log_file = open('output.txt', 'w+')
    data_procesing = open('train_dev_test_data_processing.txt', 'w+')
    data_path = "SemEval-PIT2015/data/SemEval-PIT2015-github/data"
    data_file_list = ['train', 'dev', 'test']
    # for samples number
    train_samples = 2000
    dev_samples = 200
    test_samples = 200
    # for threshold
    baseline_algo_threshold = 0.7
    TF_IDF_cosine_similarity_threshold = 0.7
    word_to_vector_cosine_similarity_threshold = 0.98
    # for binary classifier
    train_input, train_label, dev_input, dev_label, test_input, test_label = [], [], [], [], [], []
    # index and sentences mapping, {index: [sentence1, sentence2], ...}
    test_sentence_pair_index_dict = dict()
    # data processing
    for data in data_file_list:
        # calculate train input, train lable, dev input, ...
        # data_preprocessing(data)
        input_list = data + "_input"
        label_list = data + "_label"
        samples_number = str(len(eval(input_list)))
        # with open("train_dev_test_data/" + input_list, "wb") as f:   
        #     pickle.dump(eval(input_list), f)
        # with open("train_dev_test_data/" + label_list, "wb") as f: 
        #     pickle.dump(eval(label_list), f)

        # load the train input, train lable, dev input, ...
        with open("train_dev_test_data/" + input_list, "rb") as f:
            if data == "train":
                train_input = pickle.load(f)
            elif data == "dev":
                dev_input = pickle.load(f)
            elif data == "test":
                test_input = pickle.load(f)
        with open("train_dev_test_data/" + label_list, "rb") as f:
            if data == "train":
                train_label = pickle.load(f)
            elif data == "dev":
                dev_label = pickle.load(f)
            elif data == "test":
                test_label = pickle.load(f)

        samples_number = str(len(eval(input_list)))
        log_file.write(f"{data} samples: {eval(samples_number)}\n")
        data_procesing.write(f"\n{data} input: {eval(input_list)}\n{data} label: {eval(label_list)}\n\n")

    # generate the dictionary: test_sentence_pair_index_dict
    # with open("train_dev_test_data/test_sentence_pair_index_dict.pkl", "wb") as f:
    #     pickle.dump(test_sentence_pair_index_dict, f)

    # load the dictionary: test_sentence_pair_index_dict
    with open("train_dev_test_data/test_sentence_pair_index_dict.pkl", "rb") as f:
        test_sentence_pair_index_dict = pickle.load(f)

    log_file.write(f"\nCheck number of 0s and 1s labels: \ntrain data: 0s: {train_label.count(0)}, 1s: {train_label.count(1)}\
                   \ndev data: 0s: {dev_label.count(0)}, 1s: {dev_label.count(1)}\ntest data: 0s: {test_label.count(0)}, 1s: {test_label.count(1)}\n")
    print(f"test_sentence_pair_index_dict: {test_sentence_pair_index_dict}")
    # run algorithms
    # algorithm_name = ['baseline_algo', 'TF_IDF_cosine_similarity', 'word_to_vector_cosine_similarity']
    algorithm_name = ['baseline_algo', 'TF_IDF_cosine_similarity']
    feed_dev_test_into_algo()

    # only feed one feature, not good
    # for train_input_index in range(len(train_input)):
    #     train_input[train_input_index].pop(0)
    # for dev_input_index in range(len(dev_input)):
    #     dev_input[dev_input_index].pop(0)
    # for test_input_index in range(len(test_input)):
    #     test_input[test_input_index].pop(0)
    # print(test_input)

    train_dev_test_knn()
    train_dev_test_dt()

    log_file.close()
    data_procesing.close()


