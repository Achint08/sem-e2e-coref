import re
import numpy as np
import random
import collections
from keras.preprocessing.sequence import pad_sequences

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \(((..).*)\); part (\d+)")

GENRE = [ 'bc', 'bn', 'nw', 'mz', 'wb', 'tc', 'pt']

def finalize_clusters(clusters):
    merged_clusters = []
    for c1 in clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    merged_clusters = [list(c) for c in merged_clusters]
    return merged_clusters

def normalize_word(word):
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word

def get_genre(genre):
  return "{}".format(genre)

def addCharInformatioin(word):
    chars = [c for c in word]
    return chars

def readfile(filename):
    '''
    read file
    return format :
    [['genre','text','speaker']['coreference_cluster_start', 'coreference_cluster_end']]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    begin_document_match = ''
    genre = ''
    clusters = collections.defaultdict(list)
    stacks = collections.defaultdict(list)
    word_index = 0
    for line in f:
        begin_document_match = re.match(BEGIN_DOCUMENT_REGEX, line)
        if begin_document_match:
            genre = get_genre(begin_document_match.group(2))
        elif line.startswith("#end document"):
            merged_clusters = finalize_clusters(clusters)
            sentences.append([sentence, merged_clusters])
            word_index = 0;
            sentence = []
            cluster = []
        else:
            splits = line.split()
            if len(splits) >= 12:
                word = normalize_word(splits[3])
                word_index = len(sentence)
                speaker = splits[9]
                chars = addCharInformatioin(word)
                coref = splits[-1]
                if coref != '-':
                    for segment in coref.split("|"):
                        if segment[0] == "(":
                            if segment[-1] == ")":
                                cluster_id = int(segment[1:-1])
                                clusters[cluster_id].append((word_index, word_index))
                            else:
                                cluster_id = int(segment[1:])
                                stacks[cluster_id].append(word_index)
                        else:
                            cluster_id = int(segment[:-1])
                            start = stacks[cluster_id].pop()
                            clusters[cluster_id].append((start, word_index))
                sentence.append([GENRE.index(genre), word, chars, speaker])
    return sentences

def createMatrices(sentences, word2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']
        
    dataset = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        wordIndices = []
        charIndices = []
        
        for genre,word,char,speaker in sentence[0]:  
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                charIdx.append(char2Idx[x])         
            wordIndices.append(wordIdx)
            charIndices.append(charIdx)
           
        dataset.append([wordIndices, charIndices]) 

    return dataset

def padding(Sentences):
    maxlen = 50
    for sentence in Sentences:
        char = sentence[1];
    for i,sentence in enumerate(Sentences):
        Sentences[i][1] = pad_sequences(Sentences[i][1],maxlen,padding='post')
    return Sentences

def createBatches(data):
    batches = []
    batch = []
    j = 0
    for i in data:
        batch.append(i[0])
        j = j + 1
        if (j == 50):
            batches.append(batch)
            j = 0
    return batches
