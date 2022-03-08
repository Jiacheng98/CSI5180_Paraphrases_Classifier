from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from sklearn.neighbors import KNeighborsClassifier


def f1_score(micro_precision, micro_recall, precision1, recall1, precision0, recall0):
    micro_f1_score = 2 * (micro_precision * micro_recall)/(micro_precision + micro_recall)

    f1_score1 = 2 * (precision1 * recall1)/(precision1 + recall1)
    f1_score0 = 2 * (precision0 * recall0)/(precision0 + recall0)
    macro_f1_score = (f1_score1 + f1_score0)/2

    return f1_score1, f1_score0, micro_f1_score, macro_f1_score



# embed each sentence into a vector using TF_IDF (doesn't care semantics) and compute cosine similarity of the two vectors
def TF_IDF_cosine_similarity(sentence1, sentence2):
    global TF_IDF_cosine_similarity_threshold
    sentence_list = [sentence1, sentence2]
    vect = TfidfVectorizer(stop_words='english') 
    tfidf = vect.fit_transform(sentence_list)
    print(vect.get_feature_names())
    # print(vect.get_stop_words())
    
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


if __name__ == "__main__":
    micro_precision = 0.58
    micro_recall = 0.58

    precision1 = 0.47368421052631576
    recall1 = 0.21951219512195122  

    precision0 = 0.6049382716049383
    recall0 = 0.8305084745762712  

    f1_score1, f1_score0, micro_f1, macro_f1 = f1_score(micro_precision, micro_recall, precision1, recall1, precision0, recall0)
    print(f"precision1: {precision1}, recall1: {recall1}, f1_score1: {f1_score1}\
          \nprecision0: {precision0}, recall0: {recall0},, f1_score0: {f1_score0}\
          \nmicro_precision: {micro_precision}, micro_recall: {micro_recall}, micro_f1: {micro_f1} \
          \nmacro_f1: {macro_f1}")

    TF_IDF_cosine_similarity_threshold = 0.7
    TF_IDF_cosine_similarity("I got it from amazon", "According to Amazon s review")
    word_to_vector_cosine_similarity_threshold = 0.9
    word_to_vector_cosine_similarity("I got it from amazon", "According to Amazon s review")
