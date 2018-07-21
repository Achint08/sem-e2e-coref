import collections
import re
# coref = '65)|22)'
# for segment in coref.split("|"):
#       if segment[0] == "(":
#         if segment[-1] == ")":
#           cluster_id = int(segment[1:-1])
#           print cluster_id
#         else:
#           cluster_id = int(segment[1:])
#           print cluster_id
#       else:
#         cluster_id = int(segment[:-1])
#         print cluster_id

# char_vocab_path = 'char_vocab.english.txt'
# vocab = [u"<unk>"]
# char_dict = []
# with open(char_vocab_path) as f:
#   vocab.extend(str(c).strip() for c in f.readlines())
# for c in vocab:
#   print(c, end="")

def normalize_word(word):
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word
f = open("test.english.v4_gold_conll")
sentences = []
sentence = []
id = 0
begin_document_match = ''
genre = ''
clusters = collections.defaultdict(list)
stacks = collections.defaultdict(list)
word_index = 0
for line in f:
  begin_document_match = re.match(re.compile(r"#begin document \(((..).*)\); part (\d+)"), line)
  if begin_document_match:
    continue
  elif line.startswith("#end document"):
    print id
    print ' '.join(word for word in sentence)
    id = id + 1
    sentence = []
  else:
    splits = line.split()
    if len(splits) >= 12:
      word = normalize_word(splits[3])
      sentence.append(word)
