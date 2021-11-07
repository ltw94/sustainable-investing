"""
The aim of this script is to:
1. First, download all the data
2. Parse the data and clean it
3. Extract insights and append them to a dataframe.
"""
import sys
sys.path.append(".")

import pandas as pd
import json
from datetime import datetime
import spacy
from nltk.corpus import stopwords
from processingFunctions import QueryFinanceDataBase, DataReader, TextProcessor, InsightsExtractor
from tqdm import tqdm

#default variables
country = "United States"
filing_type = '10-K'
output_dir = "../data"
start_date = f"{datetime.today().year - 1}-01-01"
nlp = spacy.load("en_core_web_lg")
nlp.max_length = 1000000
english_stopwords = set(stopwords.words('english'))


def get_text_loop(query, sector):
    data_dict = {}
    df_results = QueryFinanceDataBase().query_database(query, sector, country)
    for ticker in tqdm(df_results['index']):
        reader = DataReader(ticker, filing_type, output_dir, start_date = start_date)
        download_result = reader.download_data()
        if download_result > 0:
            data = reader.read_data()
            text = reader.parse_data(data)
            processor = TextProcessor(text, nlp, english_stopwords)
            doc = processor.clean_text()
            data_dict[ticker] = doc
        else:
            continue

    return df_results, data_dict

def extract_insights(df_results, data_dict):
    extractor = InsightsExtractor(list(data_dict.values()))
    lda_model, doc_lda = extractor.evaluate_lda(num_topics=10)
    topics_df = extractor.get_topics_summary(lda_model, doc_lda)
    topics_df.loc[:, 'index'] = list(data_dict.keys())
    df_final = df_results.merge(topics_df, how = 'left')
    return df_final

def main(query, sector):
    df_results, data_dict = get_text_loop(query, sector)
    df_final = extract_insights(df_results, data_dict)
    df_final.to_csv('../output/results.csv', index=False)

