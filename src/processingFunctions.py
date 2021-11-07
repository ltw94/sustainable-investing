import pandas as pd
import financedatabase as fd
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import glob
import spacy
import string

import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora


class QueryFinanceDataBase:
    """The aim of this is to query through the financial database provided by JerBouma in the wrapped python
    library financeDatabase. In general, the database incldues 300,000+ symbols containing
    Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies and Money Markets.
    For more details, see: https://github.com/JerBouma/FinanceDatabase."""

    def query_database(self, query, sector, country):
        database = fd.select_equities(sector=sector)
        results = fd.search_products(database, query)
        results = {k: v for k, v in results.items() if v['country'] == country}

        df_results = pd.DataFrame(results).transpose().reset_index()

        return df_results

class DataReader:
    """
    The aim of this class is to read the data given an input of a ticker symbol.
    """

    def __init__(self, ticker_input, filing_type, output_dir, start_date):
        self.ticker_input = ticker_input
        self.filing_type = filing_type
        self.output_dir = output_dir
        self.start_date = start_date

    def download_data(self):
        """Downloads data in the defined directory for us to read the filing data."""
        dl = Downloader(self.output_dir)
        result = dl.get(self.filing_type, self.ticker_input, after=self.start_date)
        return result

    def read_data(self):
        """
        Reads the downloaded data to parse through the different documents downlaoaded.
        It first selects the latest year and then takes this to read the appropriate document.
        """

        directories = glob.glob(f"{self.output_dir}/sec-edgar-filings/{self.ticker_input}/{self.filing_type}/*")
        years = [dir_.split('/')[-1].split('-')[-2] for dir_ in directories]
        latest_year = sorted(years)[-1]  # select latest year from list
        idx = years.index(latest_year)
        latest_filing = directories[idx]

        with open(f"{latest_filing}/full-submission.txt") as f:
            data = f.read()

        return data

    def parse_data(self, data):
        soup = BeautifulSoup(data, 'lxml')
        filing_documents = soup.find_all('document')
        for doc in filing_documents:
            doc_id = doc.type.find(text=True, recursive=False).strip()
            if doc_id == '10-K':
                text = doc.find('text').extract()
                break

        results = text.find_all("div")

        all_text = []
        for elements in results:
            all_text.append(str(elements.get_text(strip=True)))

        final_doc = ' '.join(all_text)
        if len(final_doc) > 1000000:
            final_doc = final_doc[:1000000]
        if 'UNITED STATES SECURITIES AND EXCHANGE COMMISSION' in final_doc:
            final_doc = final_doc[final_doc.index('UNITED STATES SECURITIES AND EXCHANGE COMMISSION'):]

        return final_doc

class TextProcessor:
    """
    The aim of this Class is to process the received text so that key insights can be processed.
    """

    def __init__(self, text, nlp, stop_words):
        self.text = text
        self.nlp = nlp
        self.stop_words = stop_words

    def clean_text(self):
        punctuation = string.punctuation + '’'

        doc = self.nlp(self.text)
        clean_text = []
        for token in doc:
            if token.pos_ != 'PRON' and token.pos_ != 'PROPN' and token.lemma_ not in self.stop_words \
                    and not token.is_space and token.lemma_ not in punctuation and not token.like_num and token.is_alpha:
                clean_text.append(token.lemma_.lower())

        return ' '.join(clean_text)


class InsightsExtractor:

    def __init__(self, doc_list):
        self.doc_list = doc_list

    def evaluate_lda(self, num_topics):
        corpus_list = [x.split(' ') for x in self.doc_list]
        id2word = corpora.Dictionary(corpus_list)
        id2word.filter_extremes(no_above=0.5)
        freq_corpus = [id2word.doc2bow(text) for text in corpus_list]

        lda_model = gensim.models.LdaMulticore(corpus=freq_corpus,
                                               id2word=id2word,
                                               num_topics=num_topics)

        doc_lda = lda_model[freq_corpus]

        return lda_model, doc_lda

    def get_topics_summary(self, lda_model, doc_lda):

        df = pd.DataFrame()

        for i, row_list in enumerate(doc_lda):
            row = row_list[0] if lda_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:
                    wp = lda_model.show_topic(topic_num)
                    topic_keywords = ', '.join([word for word, prop in wp])
                    df = df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break

        df.columns = ['dominant_topic', 'perc_contribution', 'topic_keywords']

        return df

