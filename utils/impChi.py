from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Chi2TextFeatureSelector(BaseEstimator, TransformerMixin):
     """
     This class applies TF-IDF (Term Frequency-Inverse Document Frequency) to text data
     and selects the most important features (words/n-grams) for each class using the 
     Chi-Square (Chi2) statistical test. It uses a One-vs-Rest strategy. (It was also applied in the DSMLL course by me)
     """
     
     def __init__(self, 
                    text_col: str = 'TEXT', 
                    k_per_label: int = 50, 
                    min_df: int = 3,  
                    ngram_range: tuple = (1, 2)):
          
          self.text_col = text_col
          # k_per_label: How many top features to select for each unique class.
          self.k_per_label = k_per_label
          # min_df: Minimum Document Frequency. Ignores words that appear in fewer than 'min_df' documents.
          self.min_df = min_df
          # ngram_range: (1, 2) means we extract single words (unigrams) and two-word phrases (bigrams).
          self.ngram_range = ngram_range
          
          # Attributes that will be learned during the fit()
          self.tfidf_vectorizer_ = None
          self.selected_indices_ = None 
          self.feature_names_ = None
          
     def fit(self, X: pd.DataFrame, y) -> 'Chi2TextFeatureSelector':
          """
          Learns the vocabulary from the text and selects the best features based on the Chi2 test.
          """
          # I absolutely need the labels to compute the Chi-Square statistical test!
          if y is None:
               raise ValueError("Target variable 'y' is required to compute Chi2.")
          
          y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
          
          text_data = X[self.text_col].fillna('').astype(str)
          
          print(f"Fitting TF-IDF (min_df={self.min_df}, ngrams={self.ngram_range})...")
          
          self.tfidf_vectorizer_ = TfidfVectorizer(
               input='content', encoding='utf-8', lowercase=True,
               stop_words=None, # I want to keep stop words because they might be important for classification (e.g., "not", "but", "and") 
               min_df=self.min_df, 
               ngram_range=self.ngram_range
          )
          
         
          # Transform the text into a sparse matrix of TF-IDF scores
          X_tfidf = self.tfidf_vectorizer_.fit_transform(text_data)
          print(f"Selecting top {self.k_per_label} features per label via Chi-Square test...")
          unique_classes = np.unique(y_arr)
          
          feature_to_labels = {}

          # Computing Chi-Square for each class using the "One-vs-Rest" approach
          for label in unique_classes:
               # 1 if it's the current class, 0 otherwise
               y_binary = (y_arr == label).astype(int)
               
               # Compute the Chi2 scores between all TF-IDF features and the binary target
               # The chi2() function returns two arrays: scores and p-values. I only care about the scores.
               chi2_scores, _ = chi2(X_tfidf, y_binary)
               
               # Get the total number of available features in the TF-IDF vocabulary
               n_features = X_tfidf.shape[1]
               
               # Ensure we don't try to select more features than actually exist
               k_safe = min(self.k_per_label, n_features)

               if k_safe > 0:
                    top_k_indices = np.argsort(chi2_scores)[-k_safe:]
                    for idx in top_k_indices:
                         if idx not in feature_to_labels:
                              feature_to_labels[idx] = set()
                              # Record that this specific word index is a strong predictor for the current 'label'
                         feature_to_labels[idx].add(label)

          # Finalizing the selected indices and create descriptive column names
          # Extract all unique indices selected across all classes and sort them
          self.selected_indices_ = sorted(list(feature_to_labels.keys()))
          
          if not self.selected_indices_:
               print("Warning: No features were selected.")
               self.feature_names_ = []
               return self
               
          # Get the actual string words/n-grams from the fitted TF-IDF vocabulary
          raw_feature_names = self.tfidf_vectorizer_.get_feature_names_out()
          self.feature_names_ = []
          
          for idx in self.selected_indices_:
               # Extract the actual word corresponding to the numerical index
               word = raw_feature_names[idx]
               
               # Create a string suffix of the labels that this word helps predict (e.g., "0_3")
               labels_suffix = "_".join(sorted([str(lbl) for lbl in feature_to_labels[idx]]))
               
               # Construct the final descriptive feature name (e.g., "tfidf_apple_L0_3")
               self.feature_names_.append(f"tfidf_{word}_L{labels_suffix}")
               
          print(f"Total unique TF-IDF features selected: {len(self.selected_indices_)}")
          return self
     
     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
          """
          Applies the learned TF-IDF transformation and filters the matrix to keep 
          only the features selected by the Chi2 test during the fit() phase.
          """
          # Check if the model has been fitted properly
          if self.tfidf_vectorizer_ is None:
               raise RuntimeError("The Transformer is not fitted yet. Call fit() before transform().")

          # Create a copy to avoid altering the original dataframe in memory
          df = X.copy()
          text_data = df[self.text_col].fillna('').astype(str)
          
          # Initialize a list of dataframes to concatenate at the end.
          # We start by removing the original raw text column so it doesn't get passed to the ML model.
          dfs_to_concat = [df.drop(columns=[self.text_col], errors='ignore')]
          
          if len(self.selected_indices_) > 0:
               # Transform the new text data using the ENTIRE vocabulary learned during fit()
               X_tfidf_full = self.tfidf_vectorizer_.transform(text_data)
               
               # Feature Selection (Filtering)
               # Slice the sparse matrix: keep ONLY the columns (indices) selected by the Chi2 test
               X_tfidf_sel = X_tfidf_full[:, self.selected_indices_]
               
               # Convert the sparse matrix into a dense Pandas DataFrame
               df_tfidf = pd.DataFrame(
                    X_tfidf_sel.toarray(),         # .toarray() converts the sparse matrix to a dense NumPy array
                    columns=self.feature_names_,   # Apply the descriptive column names we generated in fit()
                    index=df.index                 # Keep the original index to ensure alignment with other features
               )
               
               # Add the new TF-IDF numerical dataframe to our concatenation list
               dfs_to_concat.append(df_tfidf)
               
          # Concatenate horizontally (axis=1) to combine any existing features with the new text features
          final_df = pd.concat(dfs_to_concat, axis=1)
          return final_df