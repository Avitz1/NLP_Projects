import math
import time
import random
from tokenizer import *

class NgramModel:
    def __init__(self, corpus_dir, n):
        self.n = n
        self.context_counts = {}
        self.vocab = set()
        self.sentences = []
        self.corpus_dir = corpus_dir
        self.build()

    def build(self):
        corpus = Corpus(corpus_dir)
        self.sentences = corpus.list_sentences()
        for sentence in self.sentences:
            for i in range(1 if self.n == 1 else 0, len(sentence)):
                if i == 1 and sentence[1].isalpha():
                    sentence[1] = sentence[1].lower()
                token = sentence[i]
                self.vocab.add(token)
                context = (tuple(sentence[max(0, i - self.n + 1):i]), sentence[i])
                if context not in self.context_counts:
                    self.context_counts[context] = 1
                else:
                    self.context_counts[context] += 1

    def random_token(self, context):
        reduced_to_context = {k: v for k, v in self.context_counts.items() if k[0] == context}
        reduced_to_context = [k for k, v in reduced_to_context.items() for i in range(v)]
        if len(reduced_to_context) != 0:
            return reduced_to_context[random.randint(0, len(reduced_to_context)-1)][1]
        else:
            return []

    def random_text(self, token_count, context):
        text = []
        opening_quotes = 1
        for i in range(token_count):
            next_token = self.random_token(context)
            if next_token == []:
                if text[0] and text[0][0].isalpha():
                    text[0] = text[0][0].upper() + text[0][1:]
                return ' '.join(text)
            if len(text) and len(text[-1]) and len(next_token):
                if next_token == "{" or next_token == "[" or next_token == "(":
                    text.append(next_token)
                elif next_token == ")" or next_token == "]" or next_token == "}":
                    text[-1] = text[-1] + next_token
                elif next_token == ".":
                    text[-1] = text[-1] + next_token
                elif next_token == "\"":
                    if opening_quotes:
                        text.append(next_token)
                        opening_quotes = 0
                    else:
                        text[-1] = text[-1] + next_token
                        opening_quotes = 1
                elif (text[-1][-1] == "'" and next_token == "s") or text[-1][-1] == "-" or text[-1][-1] == "–" or (not opening_quotes and text[-1][-1] == "\"") \
                        or (next_token[0] in string.punctuation and next_token != "$" and len(next_token)==1 and text[-1][-1] not in string.punctuation):
                    text[-1] = text[-1] + next_token
                elif len(text) != 0 and len(next_token) > 0 and (text[-1] == "{" or text[-1] == "[" or text[-1] == "("):
                    text[-1] = text[-1] + next_token
                else:
                    text.append(next_token)
            else:
                text.append(next_token)
            if self.n > 1:
                context = context[len(context)-1:] + tuple([next_token]) if self.n == 3 else tuple([next_token])
        if text[0][0].isalpha():
            text[0] = text[0][0].upper() + text[0][1:]
        return ' '.join(text)

    def likelihood(self, sentence, Laplace=True):
        likelihood = 1
        sentence = Sentence(sentence, 0).list_tokens()
        for i in range(1, len(sentence)):
            token = sentence[i]
            context = tuple(sentence[max(0, i - self.n + 1):i])
            likelihood *= self.prob(token, context, Laplace)
        return '{:.4f}'.format(0 if likelihood==0 else math.log(likelihood))

    def prob(self, token, context, Laplace=True):
        reduced_to_context = {k: v for k, v in self.context_counts.items() if k[0] == context}
        sample_space = 0
        for k, v in reduced_to_context.items():
            sample_space += v
        # reduced_to_context = [k for k, v in reduced_to_context.items() for i in range(v)]
        if Laplace:
            if (context, token) in self.context_counts:
                return (1 + self.context_counts[(context, token)]) / (sample_space + len(self.vocab))
            else:
                return 1 / (sample_space + len(self.vocab))
        else:
            if (context, token) in self.context_counts:
                return (self.context_counts[(context, token)]) / sample_space
            else:
                return 0


class Solution:
    def __init__(self, corpus_dir, lambda3=0.5, lambda2=0.35, lambda1=0.15):
        self.unigram = NgramModel(corpus_dir, 1)
        self.bigram = NgramModel(corpus_dir, 2,)
        self.trigram = NgramModel(corpus_dir, 3)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def getLengthFromPDF(self):
        return len(self.unigram.sentences[random.randint(0, len(self.unigram.sentences))])

    def linear_interpolation_likelihood(self, sentence):
        likelihood = 1
        sentence = sentence.split()
        for j in range(len(sentence)):
            token = sentence[j]
            likelihood *= ((self.lambda1 * self.unigram.prob(token, tuple())) +
                           (self.lambda2 * self.bigram.prob(token, tuple(sentence[max(0, j - 1):j]))) +
                           (self.lambda3 * self.trigram.prob(token, tuple(sentence[max(0, j - 2):j]), Laplace=False)))
        return '{:.4f}'.format(0 if likelihood==0 else math.log(likelihood))

if __name__ == "__main__":

    corpus_dir = argv[1]    # The directory in which the Wiki files are, full pathname
    output_file = argv[2]   # The text file that the corpus is written onto, full pathname

    models = Solution(corpus_dir)

    sentences = ["May the Force be with you.", "I’m going to make him an offer he can’t refuse.",
                 "Ogres are like onions.", "You’re tearing me apart, Lisa!", "I live my life one quarter at a time."]
    pred_models = {"Unigram Model:": models.unigram, "Bigram Model:": models.bigram, "Trigram Model:": models.trigram,}
    gen_models = {"Unigram Model:": models.unigram, "Bigram Model:": models.bigram, "Trigram Model:": models.trigram}

    with open(output_file, 'w', encoding='utf8') as f:
        sys.stdout = f

        print("\n*** Sentence Predictions ***\n\n")

        for model in pred_models:
            print("\n" + model + "\n")
            for sentence in sentences:
                if model == "Trigram Model:":
                    print(sentence + "\n" + models.linear_interpolation_likelihood(sentence))
                else:
                    print(sentence + "\n" + pred_models[model].likelihood(sentence))

        print("\n\n\n*** Random Sentence Generation ***\n\n")

        for model in gen_models:
            print("\n" + model + "\n")
            context = () if model == "Unigram Model:" else ('<s>',)
            for i in range(15):
                print(gen_models[model].random_text(models.getLengthFromPDF(), context))
        sys.stdout = sys.__stdout__
        f.close()