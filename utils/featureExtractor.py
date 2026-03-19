import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class featureExtractor(BaseEstimator, TransformerMixin):
     def __init__(self):
          pass

     @staticmethod
     def __sentence_metrics(text):
          sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 0]
          if not sentences:
               return 0, 0, 0, 0
          words_per_sentence = [len(s.split()) for s in sentences]
          avg   = np.mean(words_per_sentence)
          std   = np.std(words_per_sentence)
          burst = (std - avg) / (std + avg + 1e-9)
          return len(sentences), avg, std, burst

     @staticmethod
     def __hapax_rate(words):
          if not words:
               return 0
          c = Counter(w.lower() for w in words)
          return sum(1 for v in c.values() if v == 1) / len(words)

     def fit(self, X, y=None):
          return self

     def transform(self, X):
          df       = X.copy()
          txt      = df['TEXT'].astype(str)
          words_l  = txt.str.split()
          len_s    = txt.apply(len).replace(0, 1)
          words_s  = words_l.apply(lambda x: len(x) if isinstance(x, list) else 0).replace(0, 1)

          # length
          df['len']       = txt.apply(len)
          df['num_words'] = words_l.apply(lambda x: len(x) if isinstance(x, list) else 0)

          # Textual flags
          df['starts_lower']     = txt.apply(lambda x: x[0].islower() if x else False).astype(int)
          df['contains_genitive']= txt.str.contains("'s").astype(int)

          # Punctuation
          for name, char in [('periods','.'),('commas',','),('dashes','-'),
                              ('question','?'),('exclamation','!'),('semicolon',';'),
                              ('colon',':'),('spaces',' '),('newlines','\n'),('asterisks','*')]:
               df[f'{name}_per_len'] = txt.apply(lambda x: x.count(char)) / len_s

          df['parenthesis_per_len'] = txt.apply(lambda x: x.count('(') + x.count(')')) / len_s
          df['quotes_per_len']      = txt.apply(lambda x: x.count('"') + x.count("'") + x.count('`')) / len_s
          df['uppercase_per_len']   = txt.apply(lambda x: len(re.findall(r'[A-Z]', x))) / len_s
          df['digits_per_len']      = txt.apply(lambda x: len(re.findall(r'\d', x))) / len_s

          # Vocabulary
          df['unique_words_per_words'] = words_l.apply(
               lambda x: len(set(x)) / max(len(x), 1) if isinstance(x, list) else 0
          )
          df['avg_word_length'] = words_l.apply(
               lambda x: np.mean([len(w) for w in x]) if isinstance(x, list) and x else 0
          )
          df['hapax_rate'] = words_l.apply(
               lambda x: self.__hapax_rate(x) if isinstance(x, list) else 0
          )

          # phrase structure
          df['enumeration_num_per_len'] = txt.apply(
               lambda x: len(re.findall(r'\d+\.', x))
          ) / len_s

          metrics = txt.apply(self.__sentence_metrics)
          df['sentences_per_len']      = [m[0] for m in metrics] / len_s
          df['avg_words_per_sentence'] = [m[1] for m in metrics]
          df['std_sentence_length']    = [m[2] for m in metrics]
          df['burstiness']             = [m[3] for m in metrics]
          df['cv_sentence_length']     = df['std_sentence_length'] / (
               df['avg_words_per_sentence'].replace(0, 1)
          )

          return df

     def fit_transform(self, X, y=None, **fit_params):
          return super().fit_transform(X, y, **fit_params)