from tokenizer import *
import math
import time
import random
# Define your classes here #

class NgramModel:
    def __init__(self, corpus_dir, n, add_begin_end_tokens=True, Laplace=True):
        self.n = n
        self.context_counts = {}
        self.vocab = set()
        self.sentences = []
        self.Laplace = Laplace
        self.add_begin_end_tokens = add_begin_end_tokens
        self.corpus_dir = corpus_dir
        self.build()
        self.laplace = Laplace

    def build(self):
        corpus = Corpus(corpus_dir, self.add_begin_end_tokens)
        self.sentences = corpus.list_sentences()
        for sentence in self.sentences:
            for i in range(1 if self.n == 1 else 0, len(sentence)):
                # if i == 1:
                #     if sentence[i] and sentence[i][0] not in string.punctuation:
                #         sentence[i] = sentence[i][0].lower() + sentence[i][1:]

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

    def random_text(self, token_count, context=tuple()):
        text = []
        opening_quotes = 1
        for i in range(1, token_count):
            next_token = self.random_token(context)
            if next_token == []:
                return ' '.join(text)
            if len(text) and len(next_token) and len(text[-1]):
                if next_token[0] == "{" or next_token[0] == "[" or next_token[0] == "(":
                    text.append(next_token)
                elif next_token[0] == '.':
                    text[-1] = text[-1] + next_token
                elif next_token[0] == "\"":
                    if opening_quotes:
                        text.append(next_token)
                        opening_quotes = 0
                    else:
                        text[-1] = text[-1] + next_token
                        opening_quotes = 1
                elif text[-1][-1] == "'" or text[-1][-1] == "-" or text[-1][-1] == "–" or text[-1][-1] == "\"" or (next_token[0] in string.punctuation and len(next_token)==1 and text[-1][-1] not in string.punctuation):
                    text[-1] = text[-1] + next_token
                elif len(text) != 0 and len(next_token) > 0 and (text[-1] == "{" or text[-1] == "[" or text[-1] == "("):
                    text[-1] = text[-1] + next_token
                else:
                    text.append(next_token)
            else:
                text.append(next_token)
            if self.n > 1:
                context = context[len(context)-1:] + tuple([next_token]) if self.n == 3 else tuple([next_token])
        return ' '.join(text)

    def likelihood(self, sentence):
        likelihood = 1
        sentence = Sentence(sentence, self.add_begin_end_tokens).list_tokens()
        for i in range(1, len(sentence)):
            token = sentence[i]
            context = tuple(sentence[max(0, i - self.n + 1):i])
            likelihood *= self.prob(token, context)
        return math.log(likelihood)

    def prob(self, token, context):
        reduced_to_context = {k: v for k, v in self.context_counts.items() if k[0] == context}
        reduced_to_context = [k for k, v in reduced_to_context.items() for i in range(v)]
        if self.laplace:
            if (context, token) in self.context_counts:
                return (1 + self.context_counts[(context, token)]) / (len(reduced_to_context) + len(self.vocab))
            else:
                return 1 / (len(reduced_to_context) + len(self.vocab))
        else:
            if (context, token) in self.context_counts:
                return (self.context_counts[(context, token)]) / (len(reduced_to_context))
            else:
                return 1e-25


class Solution:
    def __init__(self, corpus_dir, lambda3 = 0.5, lambda2 = 0.3, lambda1 = 0.2, add_begin_end_tokens=True, Laplace=True):
        self.unigram = NgramModel(corpus_dir, 1, add_begin_end_tokens, Laplace)
        self.bigram = NgramModel(corpus_dir, 2, add_begin_end_tokens, Laplace)
        self.trigram = NgramModel(corpus_dir, 3, add_begin_end_tokens, Laplace=False)
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
                           (self.lambda3 * self.trigram.prob(token, tuple(sentence[max(0, j - 2):j]))))
        return math.log(likelihood)

if __name__ == "__main__":

    start_time = time.time()
    # corpus_dir = argv[1]    # The directory in which the Wiki files are, full pathname
    # output_file = argv[2]   # The text file that the corpus is written onto, full pathname
    output_file = "generated.txt"
    corpus_dir = "corpus_dir"
    calc_prob = Solution(corpus_dir)
    generator = Solution(corpus_dir, Laplace=False)
    i = calc_prob.unigram.likelihood("it is")
    print('{:.4f}'.format(i))
    i = calc_prob.bigram.likelihood("it is")
    print('{:.4f}'.format(i))
    i = calc_prob.trigram.likelihood("it is")
    print('{:.4f}'.format(i))
    i = calc_prob.linear_interpolation_likelihood("it is")
    print('{:.4f}'.format(i))
    length = generator.getLengthFromPDF()
    print("the required length was " + str(length))
    with open(output_file, 'w', encoding='utf8') as f:
        sys.stdout = f
        print("now unigrams:")
        for i in range(10):
            print(generator.unigram.random_text(length))
        print("now bigrams:")
        for i in range(10):
            print(generator.bigram.random_text(length, ('<s>',)))
        print("now trigrams:")
        for i in range(10):
            print(generator.trigram.random_text(length, ('<s>',)))
        sys.stdout = sys.__stdout__
        f.close()
    print("--- %s seconds ---" % (time.time() - start_time))

