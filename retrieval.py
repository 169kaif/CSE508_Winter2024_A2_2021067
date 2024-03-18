# make imports
from io import BytesIO
import os
import numpy as np
import cosine_sim as Cosim
import requests
import string
import pickle
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import PIL
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import csv
from torchvision.models import resnet50, ResNet50_Weights
from matplotlib import image,pyplot

# initialize cosine similarity object
cosim = Cosim.Cosim()

# read the csv file
def read_csv(file_name):
    
    # check if the file exists
    if os.path.exists(file_name):

        # open the file
        with open(file_name, 'r') as file:

            # read the file
            reader = csv.reader(file)

            # return the data
            return list(reader)
        
    return None

num_image_review = read_csv('A2_Data.csv')
num_image_review_dict = dict()

for i in range(1, len(num_image_review)):
    new_id = int(num_image_review[i][0])
    num_image_review_dict[new_id] = []
    num_image_review_dict[new_id].append(num_image_review[i][2])

    image_links = num_image_review[i][1][1:-1].split(',')
    for i in range(len(image_links)):
        image_links[i] = image_links[i].strip()[1:-1]

    num_image_review_dict[new_id].append(image_links)

# download the nltk punkt corpus for tokenization
nltk.download('punkt')

# download the stopword corpus to get rid of stopwords later
nltk.download('stopwords')

# download wordnet lemmatizer
nltk.download('wordnet')

# import word_to_idx dict
with open('word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

# import idx_to_word dict
with open('idx_to_word.pkl', 'rb') as f:
    idx_to_word = pickle.load(f)

# import idx_to_doc dict
with open('idx_to_doc.pkl', 'rb') as f:
    idx_to_doc = pickle.load(f)

# import idf_dict dict
with open('idf_dict.pkl', 'rb') as f:
    idf_dict = pickle.load(f) 

# import tfidf_mat.pkl
with open('tfidf_mat.pkl', 'rb') as f:
    tfidf_mat = pickle.load(f)

# import extracted image features and labels
with open('image_features.pkl', 'rb') as f:
    all_image_features = pickle.load(f)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+') 
    return re.sub(pattern, r'', text)

def remove_html(text):
    pattern = re.compile('<.*?>')
    return re.sub(pattern, r'', text)

doc_to_idx = dict()
for idx, doc in idx_to_doc.items():
    doc_to_idx[doc] = idx

num_unique_words = tfidf_mat.shape[0]
num_docs = tfidf_mat.shape[1]

while True:

    link_input = input("Enter Image URL: ")
    review_input = input("Enter review text: ")

    print("\n\n\n")

    # preprocess the input text

    # get rid of links
    review_input = remove_url(review_input)

    # get rid of html classes
    review_input = remove_html(review_input)

    # remove punctuation
    review_input = "".join([i for i in review_input if i not in string.punctuation])

    # make it lowercase
    review_input = review_input.lower()

    # tokenize the text
    review_input = word_tokenize(review_input)

    # remove stopwords
    stop_words = set(stopwords.words('english'))

    ri_wo_sw = []
    for word in review_input:
        if word not in stop_words:
            ri_wo_sw.append(word)

    review_input = ri_wo_sw

    # apply lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    review_input = [wordnet_lemmatizer.lemmatize(word) for word in review_input]

    # apply stemming
    porter_stemmer = PorterStemmer()
    review_input = [porter_stemmer.stem(word) for word in review_input]

    # make tfidf vector for review_input
    tf_review = np.zeros(num_unique_words)

    for word in review_input:
        if word in word_to_idx:
            tf_review[word_to_idx[word]] += 1

    # calculate sum of tf_review
    tf_review_sum = 0
    for i in range(num_unique_words):
        tf_review_sum += tf_review[i]

    tf_review /= (tf_review_sum + 1e-5)

    #multiply with idf
    for i in range(num_unique_words):
        tf_review[i] *= idf_dict[i]

    # tf-idf vector for input review is given by tf_review
        
    # download image from input url
    dwnld_image = requests.get(link_input).content
    curr_input_image = Image.open(BytesIO(dwnld_image))
    curr_input_image = curr_input_image.resize((224,224))
    input_image = np.asarray(curr_input_image)

    input_image_copy = input_image.copy()

    # define and apply the transform
    transform = transforms.Compose([
    transforms.ToTensor(),  # Convert numpy array to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize if needed
    ])

    image_tensor = transform(input_image_copy)

    # add extra dimension for the batch
    image_tensor = image_tensor.unsqueeze(0)

    # load pre-trained ResNet model
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # remove the last fully connected layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    # set the model to evaluation mode
    resnet.eval()

    with torch.no_grad():
        input_image_features = resnet(image_tensor)
        input_image_features = input_image_features.squeeze()

    # normalize input image features according to the features of the image
        
    normalized_features_np = input_image_features.numpy()
    normalized_features_np = (normalized_features_np-np.mean(normalized_features_np))/ (np.std(normalized_features_np)+(1e-5))

    # init list to store composite scores w/ ele = [composite_score, product_id, image_link]
    composite_scores = []

    # text based similarity
    text_sim_scores = []
    for i in range(num_docs):
        text_sim_scores.append(cosim.sim(tf_review, tfidf_mat[:,i]))

    # rank the text similarity scores
    text_sim_scores = np.array(text_sim_scores)
    text_scores_idxs = np.argsort(text_sim_scores)[::-1]

    print("Top 5 documents retrieved based on text similarity: ")
    for i in range(5):
        print("Rank: ", i+1)
        print("Product ID: ", idx_to_doc[text_scores_idxs[i]])
        print("Text Cosine Similarity: ", text_sim_scores[text_scores_idxs[i]])
        print("Review Text: ", num_image_review_dict[idx_to_doc[text_scores_idxs[i]]][0])    
        print("\n\n")

        tr_image_link = num_image_review_dict[idx_to_doc[text_scores_idxs[i]]][1][0]
        print("Image Link: ", tr_image_link)
        for j in range(len(all_image_features)):
            if all_image_features[j][2] == tr_image_link:
                image_cosine_sim = cosim.sim(normalized_features_np, all_image_features[j][3])
                print("Image Cosine Similarity: ", image_cosine_sim)
                comp_ret_score = ( (text_sim_scores[text_scores_idxs[i]]) + (image_cosine_sim) ) / 2
                composite_scores.append([comp_ret_score, idx_to_doc[text_scores_idxs[i]], tr_image_link])
                break
        print("\n\n")

    # image based similarity
    image_sim_scores = []
    for i in range(len(all_image_features)):
        image_sim_scores.append(cosim.sim(normalized_features_np, all_image_features[i][3]))

    # rank the image similarity scores
    image_sim_scores = np.array(image_sim_scores)
    image_scores_idxs = np.argsort(image_sim_scores)[::-1]

    docs_found = set()

    print("Top 5 documents retrieved based on image similarity: ")
    docs_retrieved = 0
    curr_idx = 0

    while docs_retrieved < 5:

        product_id = int(all_image_features[image_scores_idxs[curr_idx]][1])

        if product_id in docs_found:
            curr_idx += 1
            continue
        else:
            docs_found.add(product_id)
            docs_retrieved += 1

        print("Rank: ", docs_retrieved)
        print("Product ID: ", product_id)
        print("Image Link: ", all_image_features[image_scores_idxs[curr_idx]][2])
        print("Image Cosine Similarity: ", image_sim_scores[image_scores_idxs[curr_idx]])
        print("\n\n")

        print("Review Text: ", num_image_review_dict[product_id][0])
        print("Text Cosine Similarity: ", text_sim_scores[doc_to_idx[product_id]])
        print("\n\n")

        comp_ret_score = ( (text_sim_scores[doc_to_idx[product_id]]) + (image_sim_scores[image_scores_idxs[curr_idx]]) ) / 2
        composite_scores.append([comp_ret_score, product_id, all_image_features[image_scores_idxs[curr_idx]][2]])

    print("Top documents retrieved based on composite similarity: ")
    composite_scores = sorted(composite_scores, key=lambda x: x[0], reverse=True)
    for i in range(len(composite_scores)):
        print("Rank: ", i+1)
        print("Product ID: ", composite_scores[i][1])
        print("Review Text: ", num_image_review_dict[composite_scores[i][1]][0])
        print("Image Link: ", composite_scores[i][2])
        print("Composite Similarity: ", composite_scores[i][0])
        print("\n\n")