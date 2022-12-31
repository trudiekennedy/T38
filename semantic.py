import spacy
nlp = spacy.load('en_core_web_md')

# === Similarity with spaCy ===

word1 = nlp('cat')
word2 = nlp('monkey')
word3 = nlp('banana')

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# === Working with vectors ===
print("")
tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

""" It's interesting that 'apple' and 'banana' (fruits) have a higher similarity than 'monkey' and 'cat' (animals).
Cat has very little similarity with any of the fruit words; probably because they're not known for eating fruits!
Monkey has more of a similarity with the fruit words than cat does. 
"""

# === Own examples ===
print("")
tokens = nlp('red yellow blue green')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

""" 'red' and 'green' has the lowest similarity - I think this means that the model recognises that green is not a 
primary colour and red is not involved in making green (yellow and green & blue and yellow have a 
slightly higher similarity
"""


# === Working with sentences ===
print(f"")
sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
