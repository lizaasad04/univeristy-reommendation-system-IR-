import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from fuzzywuzzy import fuzz
import string
from sentence_transformers import SentenceTransformer


class AustralianUniversitiesIR:
    def __init__(self, file_path):
        self.file_path = file_path
        # ins,courses,locs,and course_loc will store DataFrames for the respective sheets in the Excel file.
        self.institutions_df = None
        self.courses_df = None
        self.locations_df = None
        self.course_loc_df = None
        self.merged_df = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.semantic_model = None
        self.semantic_matrix = None

    def load_data(self):
        try:
            xls = pd.ExcelFile(self.file_path,
                               engine='openpyxl')  # loads file using pandas,openpyxl is req for xlsx files(a Python library designed to read, write, and modify Excel files)
            print(
                f"Excel file loaded successfully. Available sheets: {xls.sheet_names}")  # prints all sheets names in excel file

            # load institutions data
            self.institutions_df = pd.read_excel(xls, sheet_name='institutions',
                                                 skiprows=2)  # skip first two rows (if they contain metadeta)
            self.institutions_df.columns = self.institutions_df.columns.str.strip().str.lower().str.replace(" ",
                                                                                                            "_")  # clean column names
            print(f"Institutions data loaded: {self.institutions_df.shape[0]} rows")  # prints no of rows loaded

            # load courses data
            self.courses_df = pd.read_excel(xls, sheet_name='courses', skiprows=2)
            self.courses_df.columns = self.courses_df.columns.str.strip().str.lower().str.replace(" ", "_")
            print(f"Courses data loaded: {self.courses_df.shape[0]} rows")

            # load locations data
            self.locations_df = pd.read_excel(xls, sheet_name='locations', skiprows=2)
            self.locations_df.columns = self.locations_df.columns.str.strip().str.lower().str.replace(" ", "_")
            print(f"Locations data loaded: {self.locations_df.shape[0]} rows")

            # load course locations data
            self.course_loc_df = pd.read_excel(xls, sheet_name='course loc', skiprows=2)
            self.course_loc_df.columns = self.course_loc_df.columns.str.strip().str.lower().str.replace(" ", "_")
            print(f"Course locations data loaded: {self.course_loc_df.shape[0]} rows")

            # clean dataframes
            self._clean_dataframes()  # calls method for cleaning any issues in data, defined below
            print("Data loaded successfully and cleaned")
            return True

        except Exception as e:
            print(f"error loading data:{e}")
            return False

    def _clean_dataframes(self):
        # cleaning by removing rows nd cols
        for df_name in ['institutions_df', 'courses_df', 'locations_df',
                        'course_loc_df']:  # loops through the 4 dfs as strings
            df = getattr(self,
                         df_name)  # retrieves the df from the class ka instance, same as self.institutions_df etc etc
            df.dropna(how='all', inplace=True)  # rmove cella jahan NaN ho
            df.dropna(axis=1, how='all', inplace=True)
            # remove dup rows
            df.drop_duplicates(inplace=True)
            setattr(self, df_name, df)  # save cleaned dataframe back to class attribute

    def preprocess_data(self):
        if not all([isinstance(self.institutions_df, pd.DataFrame),  # checks whether each var is a df or not
                    isinstance(self.courses_df, pd.DataFrame),
                    isinstance(self.locations_df, pd.DataFrame),
                    isinstance(self.course_loc_df, pd.DataFrame)]):  # check if all r actual pandas df
            print("Dataframes not loaded correctly")
            return False

        # SARE FUNCTIONS R DEFINED LATER IN CODE NEXT AFTER THIS FNC

        # ceating complete address , jo 4 blocks m hai usko ek m krdia
        # combines 4 separate columns into one column called full_address
        self._merge_address_columns(self.institutions_df,
                                    ['postal_address_line_1', 'postal_address_line_2', 'postal_address_line_3',
                                     'postal_address_line_4'], 'postal_address_full')

        # Same logic as above, but applied to the locations_df
        self._merge_address_columns(self.locations_df,
                                    ['address_line_1', 'address_line_2', 'address_line_3', 'address_line_4'],
                                    'full_address')
        # merge field of education columns in courses
        self._merge_field_of_education()  # a func

        # process course duration and fees
        self._process_course_details()  # a func

        print("Data preprocessing completed!")
        return True

    def _merge_address_columns(self, df, address_cols, new_col_name):
        # Check if all columns exist in the dataframe
        existing_cols = [col for col in address_cols if col in df.columns]
        if not existing_cols:
            print(f"Warning: None of the address columns {address_cols} found in dataframe")
            df[new_col_name] = ""
            return
        # Merge multiple address columns into one
        # applies a func row wise, drops NaNs. converts each value to string and strips spaces, then joins all parts with commas
        # ['123 Main St', NaN, 'Building 4', 'Room 56'],  BECOMES    "123 Main St, Building 4, Room 56"
        df[new_col_name] = df[address_cols].apply(
            lambda x: ', '.join(x.dropna().astype(str).str.strip()),
            axis=1)  # takes all add parts in row removes mpty values ans join all with commas

        # this if else is used cz in case of locations sheet city, state, postcode are col names whereas in institutions sheet postal adress city etc r used
        if all(col in df.columns for col in ['city', 'state', 'postcode']):
            df[new_col_name] = df.apply(
                lambda x: f"{x[new_col_name]}, {x['city']}, {x['state']} {x['postcode']}"
                if pd.notna(x[new_col_name]) and x[new_col_name] != ""
                else f"{x['city']}, {x['state']} {x['postcode']}",
                axis=1)

        elif all(col in df.columns for col in
                 ['postal_address_city', 'postal_address_state', 'postal_address_postcode']):
            df[new_col_name] = df.apply(
                lambda
                    x: f"{x[new_col_name]}, {x['postal_address_city']}, {x['postal_address_state']} {x['postal_address_postcode']}"
                if pd.notna(x[new_col_name]) and x[new_col_name] != ""
                else f"{x['postal_address_city']}, {x['postal_address_state']} {x['postal_address_postcode']}",
                axis=1)  # if its empty ten just return city, state and postcode
            # retuens format adress, cuty, state, postcode

    def _merge_field_of_education(self):
        # Check if necessary columns exist
        foe_cols = ['field_of_education_1_broad_field', 'field_of_education_1_narrow_field',
                    'field_of_education_1_detailed_field', 'field_of_education_2_broad_field',
                    'field_of_education_2_narrow_field', 'field_of_education_2_detailed_field']

        missing_cols = [col for col in foe_cols if col not in self.courses_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns for field of education: {missing_cols}")
            # Create empty columns for missing fields
            for col in missing_cols:
                self.courses_df[col] = None

        # First field of education
        self.courses_df['field_of_education_1'] = self.courses_df.apply(
            lambda x: ', '.join(filter(lambda y: pd.notna(y) and str(y).strip(),
                                       [str(x['field_of_education_1_broad_field']) if pd.notna(
                                           x['field_of_education_1_broad_field']) else None,
                                        # if exists(not null), else none
                                        str(x['field_of_education_1_narrow_field']) if pd.notna(
                                            x['field_of_education_1_narrow_field']) else None,
                                        str(x['field_of_education_1_detailed_field']) if pd.notna(
                                            x['field_of_education_1_detailed_field']) else None])),
            axis=1
        )
        # Second field of education (if exists)
        self.courses_df['field_of_education_2'] = self.courses_df.apply(
            lambda x: ', '.join(filter(lambda y: pd.notna(y) and str(y).strip(),
                                       [str(x['field_of_education_2_broad_field']) if pd.notna(
                                           x['field_of_education_2_broad_field']) else None,
                                        str(x['field_of_education_2_narrow_field']) if pd.notna(
                                            x['field_of_education_2_narrow_field']) else None,
                                        str(x['field_of_education_2_detailed_field']) if pd.notna(
                                            x['field_of_education_2_detailed_field']) else None])),
            axis=1
        )

        # Combine both fields of education
        self.courses_df['fields_of_education'] = self.courses_df.apply(
            lambda x: '; '.join(filter(None,
                                       [x['field_of_education_1'] if x['field_of_education_1'] else None,
                                        x['field_of_education_2'] if x['field_of_education_2'] else None])),
            axis=1
        )

    def _process_course_details(self):
        # process course duration and fees
        # column name could be either 'duration_(weeks)' or 'duration(weeks)'
        duration_col = None
        if 'duration_(weeks)' in self.courses_df.columns:
            duration_col = 'duration_(weeks)'
        elif 'duration(weeks)' in self.courses_df.columns:
            duration_col = 'duration(weeks)'

        if duration_col:
            # convert weeks to years and numeric val
            self.courses_df['duration_weeks'] = pd.to_numeric(self.courses_df[duration_col], errors='coerce')
            self.courses_df['duration_years'] = (self.courses_df['duration_weeks'] / 52).round(1)
        else:
            print("Warning: Duration column not found")

        fee_columns = ['tuition_fee', 'non_tuition_fee', 'estimated_total_course_cost']
        for col in fee_columns:
            if col in self.courses_df.columns:
                # remove currency symbols and commas nd convert to nmeric
                self.courses_df[col] = self.courses_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                self.courses_df[col] = pd.to_numeric(self.courses_df[col], errors='coerce')

        # format work component details
        work_cols = ['work_component', 'work_component_hours/week', 'work_component_weeks',
                     'work_component_total_hours']
        if all(col in self.courses_df.columns for col in work_cols):
            self.courses_df['work_component_details'] = self.courses_df.apply(
                lambda
                    x: f"Yes, {x['work_component_hours/week']} hours/week for {x['work_component_weeks']} weeks (Total: {x['work_component_total_hours']} hours)"
                if pd.notna(x['work_component']) and str(x['work_component']).lower() == 'yes'
                else "No work component",
                axis=1)
            # example otput of it: yes, 10 hours/week for 12 weeks (total:120 hours)

    # now, merging all of above into one dataset
    def merge_datasets(self):
        if not all([
            isinstance(self.institutions_df, pd.DataFrame),
            isinstance(self.courses_df, pd.DataFrame),
            isinstance(self.locations_df, pd.DataFrame),
            isinstance(self.course_loc_df, pd.DataFrame)
        ]):
            print("Dataframes not loaded correctly")
            return False

        if not self.preprocess_data():
            print("Data preprocessing failed")
            return False

        try:
            # Merge courses with course locations
            print("Merging courses with course locations...")
            courses_with_locations = pd.merge(
                self.courses_df,
                self.course_loc_df,
                on=['cricos_provider_code', 'institution_name'],
                how='left'  # ensure sare col aye
            )
            print(f"After first merge: {len(courses_with_locations)} rows")

            # Merge with location details
            print("Merging with location details...")
            if 'full_address' not in self.locations_df.columns:
                print("Warning: 'full_address' column not found in locations dataframe")

            location_cols = ['cricos_provider_code', 'location_name']
            if 'full_address' in self.locations_df.columns:
                location_cols.append('full_address')

            courses_with_full_locations = pd.merge(
                courses_with_locations,
                self.locations_df[location_cols],
                on=['cricos_provider_code', 'location_name'],
                how='left')
            print(f"After second merge: {len(courses_with_full_locations)} rows")

            # Merge with institution details
            print("Merging with institution details...")
            institution_cols = ['cricos_provider_code']
            for col in ['institution_type', 'institution_capacity', 'website', 'postal_address_full']:
                if col in self.institutions_df.columns:
                    institution_cols.append(col)
                else:
                    print(f"Warning: '{col}' column not found in institutions dataframe")

            self.merged_df = pd.merge(
                courses_with_full_locations,
                self.institutions_df[institution_cols],
                on='cricos_provider_code',
                how='left')

            self._prepare_final_dataset()
            print(
                f"Successfully merged datasets! Final dataset has {len(self.merged_df)} rows and {len(self.merged_df.columns)} columns.")
            return True

        except Exception as e:
            print(f"Error during dataset merging: {e}")
            return False

    def _prepare_final_dataset(self): #prepares a merged dataset for text-based searching using both TF-IDF and semantic embeddings.
        # Prepare the final dataset for searching
        text_cols = ['institution_name', 'course_name', 'fields_of_education', 'course_level', 'location_city',
                     'location_state', 'institution_type']
        existing_cols = [col for col in text_cols if col in self.merged_df.columns]

        self.merged_df['searchable_text'] = self.merged_df.apply( #- Creates a new column called that combines the values from these columns into a single string `searchable_text`
            lambda x: ' '.join(filter(None, [str(x[col]) if pd.notna(x[col]) else '' for col in existing_cols])),
            axis=1
        )

        self.merged_df['processed_text'] = self.merged_df['searchable_text'].apply(self._preprocess_text) #This applies the method to each row's searchable text. `_preprocess_text`

        # Remove duplicates based on key columns before creating TF-IDF matrix
        key_columns = ['institution_name', 'course_name', 'location_name']
        self.merged_df = self.merged_df.drop_duplicates(subset=key_columns)

        # Convert all processed text into TF-IDF matrix
        #initialise a tf-idf vector that include terms that exist in atleast one doc, and exclude the ones that appear in 95% of docs (too common terms)
        self.tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.merged_df['processed_text'])

        #loads a pretrained sentence transformer, which encodes all processed text into semantic embeddings, which capture meaning of the word too, not just keyword matching
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_matrix = self.semantic_model.encode(self.merged_df['processed_text'].tolist(),
                                                          show_progress_bar=True)

    def save_processed_data(self, save_path="cached_data"):
        os.makedirs(save_path, exist_ok=True)#create directory if it doesnt exist

        self.merged_df.to_pickle(os.path.join(save_path, "merged_df.pkl"))#saved merged df to a pickle file

        #- `pickle` Uses Python's library to serialize and save the TF-IDF vectorizer object
        with open(os.path.join(save_path, "tfidf_vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        with open(os.path.join(save_path, "tfidf_matrix.pkl"), 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)# - Saves the TF-IDF matrix, which contains the vectorized representation of the text data

        if self.semantic_matrix is not None: #save semantic matrix if it exists
            with open(os.path.join(save_path, "semantic_matrix.pkl"), 'wb') as f:
                pickle.dump(self.semantic_matrix, f)

        print("Processed data cached successfully.")


    def load_cached_data(self, save_path="cached_data"):
        try:
            self.merged_df = pd.read_pickle(os.path.join(save_path, "merged_df.pkl"))#Loads the previously processed and merged dataframe using pandas' pickle functionality.

            with open(os.path.join(save_path, "tfidf_vectorizer.pkl"), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)#load tf idf vectoriser
                #- The `TfidfVectorizer`contains the vocabulary, IDF values, and normalization parameters.

            with open(os.path.join(save_path, "tfidf_matrix.pkl"), 'rb') as f:
                self.tfidf_matrix = pickle.load(f)

            # this is only for , if semantic embeddings were generated and saved, if so, load their matrix
            semantic_path = os.path.join(save_path, "semantic_matrix.pkl")
            if os.path.exists(semantic_path):
                with open(semantic_path, 'rb') as f:
                    self.semantic_matrix = pickle.load(f)
                # Reload the model if not already loaded
                if self.semantic_model is None:
                    from sentence_transformers import SentenceTransformer
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

            print("Cached data loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load cached data: {e}")
            return False

    def _preprocess_text(self, text):
        # skip if not str
        if not isinstance(text, str):
            return ""
        # convert to lowercase
        text = text.lower()
        # remove punctuation ,, ., !, etc.
        text = text.translate(str.maketrans('', '', string.punctuation))
        # tokenize
        tokens = nltk.word_tokenize(text)
        # remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        # lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        # return clean string
        return ' '.join(tokens)

    #main function for searching
    def search(self, query, top_n=10, method='default'):
        if method == 'default':  # default setting is TF-IDF + cosine similarity
            if self.tfidf_matrix is None: #error checking to see if tfidf matrix exists
                print("Error: TF-IDF matrix not built. Run merge_datasets() first.")
                return None

            # preprocess query and find similarities with tfidf matrix
            processed_query = self._preprocess_text(query)
            query_vec = self.tfidf_vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_n:][::-1]
            results = self.merged_df.iloc[top_indices].copy()
            results['similarity_score'] = similarities[top_indices]
            return results

        elif method == 'semantic':
            if not hasattr(self, 'semantic_model') or not hasattr(self, 'semantic_matrix'):
                print("Error: Semantic model or semantic matrix not initialized.")
            return None

        elif method == ' y':
            if self.merged_df is None or 'searchable_text' not in self.merged_df.columns:
                print("Error: Merged data not ready. Run merge_datasets() first.")
                return None

            results = []
            for idx, row in self.merged_df.iterrows():
                text = row['searchable_text']
                score = fuzz.token_set_ratio(query.lower(), str(text).lower())
                results.append((idx, score))

            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
            indices = [idx for idx, score in results]
            scores = [score for idx, score in results]

            final_results = self.merged_df.iloc[indices].copy()
            final_results['similarity_score'] = [s / 100.0 for s in scores]  # normalize 0-1 like other methods
            return final_results

            # Encode query using semantic embedding model
            query_embedding = self.semantic_model.encode([query])
            # Compute cosine similarity between query embedding and document embeddings
            similarities = cosine_similarity(query_embedding, self.semantic_matrix).flatten()
            top_indices = similarities.argsort()[-top_n:][::-1]
            results = self.merged_df.iloc[top_indices].copy()
            results['similarity_score'] = similarities[top_indices]
            return results

        else:
            print(f"Unknown search method: {method}. Available: 'default', 'semantic'.")
            return None