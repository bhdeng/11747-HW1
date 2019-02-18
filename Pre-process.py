
# coding: utf-8

# In[ ]:


from collections import Counter

word_dict = Counter()

def count_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            topic, words = line.lower().strip().split(" ||| ")
            words = words.split(" ")
            for w in words:
                word_dict[w.lower()] += 1

count_dataset("./topicclass/topicclass_train.txt")
len(word_dict)


# In[ ]:


truncated_word_dict = defaultdict(lambda: len(truncated_word_dict))
truncated_word_dict["<unk>"]
for w in word_dict.keys():
    if word_dict[w] > 1:
        truncated_word_dict[w]
len(truncated_word_dict)


# In[ ]:


np.save("vocab.npy", dict(truncated_word_dict))

