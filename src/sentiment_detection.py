import sys
sys.path.insert(1, './')

def finalUnigramValue(review, P_init, P_dict, P_UNK):
    unigramVal = P_init

    for word in review:
        if word in P_dict:
            wordProb = P_dict[word]
        # if not found
        else:
            wordProb = P_UNK
        
        unigramVal += wordProb

    return unigramVal

def finalBigramValue(review, P_init, P_dict, P_UNK):
    bigramVal = P_init
    
    for i in range(len(review)-1):
        key = review[i] + review[i+1]
        if key in P_dict:
            wordProb = P_dict[key]
        # if not found
        else:
            wordProb = P_UNK
        
        bigramVal += wordProb

    return bigramVal

def naiveBayes(reviewStr, unigramValues):
    #parse review input to list ["hello", "world"]
    reviewList = re.findall(r'\w+', reviewStr) 

    predictPositive = finalUnigramValue(reviewList, unigramValues.P_pos, unigramValues.dict_pos, unigramValues.P_UNK_POS)
    predictNegative = finalUnigramValue(reviewList, unigramValues.P_neg, unigramValues.dict_neg, unigramValues.P_UNK_NEG)

    # if more likely to be positive
    if predictPositive > predictNegative:
        return 1

    return 0

def bigramBayes(reviewStr, unigramValues, bigramValues, bigram_lambda = 0.15):
    #parse review input to list ["hello", "world"]
    reviewList = re.findall(r'\w+', reviewStr) 
    
    predictUnigramPositive = finalUnigramValue(reviewList, unigramValues.P_pos, unigramValues.dict_pos, unigramValues.P_UNK_POS)
    predictUnigramNegative = finalUnigramValue(reviewList, unigramValues.P_neg, unigramValues.dict_neg, unigramValues.P_UNK_NEG)

    predictBigramPositive = finalBigramValue(reviewList, bigramValues.P_BI_POS, bigramValues.dict_BI_pos, bigramValues.P_UNK_BI_POS)
    predictBigramNegative = finalBigramValue(reviewList, bigramValues.P_BI_NEG, bigramValues.dict_BI_neg, bigramValues.P_UNK_BI_NEG)

    finalPositive = bigram_lambda * predictBigramPositive + (1-bigram_lambda) * predictUnigramPositive
    finalNegative = bigram_lambda * predictBigramNegative + (1-bigram_lambda) * predictUnigramNegative

    if finalPositive > finalNegative:
        return 1

    return 0

# get vals for unigram
def getUnigramParams(train_set, train_labels, smoothing_parameter = 0.6, pos_prior = 0.8):
    P_pos = np.log(pos_prior)
    P_neg = np.log(1-pos_prior)

    totalPositiveWords = 0
    totalNegativeWords = 0

    dict_pos = defaultdict(float)
    dict_neg = defaultdict(float)

    # remove stopwords
    sw = stopwords.words("english")
    for review in train_set:
        for word in review:
            if word in sw:
                review.remove(word)

    # build probability
    for i in range(len(train_set)):
        curLabel = train_labels[i]

        for word in train_set[i]:
            # if positive review, add to positive dict
            if curLabel == 1:
                dict_pos[word] += 1
                totalPositiveWords += 1
            # if negative review, add to negative dict
            else:
                dict_neg[word] += 1
                totalNegativeWords += 1
                 

    # update dictionary and get unknowns
    V_pos = len(dict_pos)
    for word in dict_pos:
        dict_pos[word] = np.log((dict_pos[word] + smoothing_parameter) / (totalPositiveWords + smoothing_parameter*(1 + V_pos)))
    P_UNK_POS = np.log((smoothing_parameter) / (totalPositiveWords + smoothing_parameter*(1 + V_pos)))

    V_neg = len(dict_neg)

    for word in dict_neg:
        dict_neg[word] = np.log((dict_neg[word] + smoothing_parameter) / (totalNegativeWords + smoothing_parameter*(1 + V_neg)))
    P_UNK_NEG = np.log((smoothing_parameter) / (totalNegativeWords + smoothing_parameter*(1 + V_neg)))

    return P_pos, P_neg, dict_pos, dict_neg, P_UNK_POS, P_UNK_NEG


def getBigramParams(train_set, train_labels, bigram_smoothing_parameter = 0.001 ,pos_prior = 0.8):
    P_UNK_BI_POS = 0
    P_UNK_BI_NEG = 0

    P_BI_POS = np.log(pos_prior)
    P_BI_NEG = np.log(1-pos_prior)

    totalBiPositive = 0
    totalBiNegative = 0

    dict_BI_neg = defaultdict(float)
    dict_BI_pos = defaultdict(float)
    
    # count occurences
    for i in range(len(train_set)):
        curLabel = train_labels[i]
        for j in range(len(train_set[i])-1):
            key = train_set[i][j] + train_set[i][j+1]
            # if positive
            if curLabel == 1:
                dict_BI_pos[key] += 1
                totalBiPositive += 1
            else:
                dict_BI_neg[key] += 1
                totalBiNegative += 1
    
    # build probability
    V_pos = len(dict_BI_pos)
    for key in dict_BI_pos:
        dict_BI_pos[key] = np.log((dict_BI_pos[key] + bigram_smoothing_parameter) / (totalBiPositive + bigram_smoothing_parameter*(1 + V_pos)))
    P_UNK_BI_POS = np.log((bigram_smoothing_parameter) / (totalBiPositive + bigram_smoothing_parameter*(1 + V_pos)))

    V_neg = len(dict_BI_neg)
    for key in dict_BI_neg:
        dict_BI_neg[key] = np.log((dict_BI_neg[key] + bigram_smoothing_parameter) / (totalBiNegative + bigram_smoothing_parameter*(1 + V_neg)))
    P_UNK_BI_NEG = np.log((bigram_smoothing_parameter) / (totalBiNegative + bigram_smoothing_parameter*(1 + V_neg)))

    return P_BI_POS, P_BI_NEG, dict_BI_pos, dict_BI_neg, P_UNK_BI_POS, P_UNK_BI_NEG