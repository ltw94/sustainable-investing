# sustainable-investing
A project leveraging the power of open-data and python to help individual investors take a data-driven approach to ESG investing. 
Specifically

## how the script works
This python script takes two inputs:
- The text you wish to query
- The sector you wish to filter through

These two inputs will filter through the equities database provided by https://github.com/JerBouma/FinanceDatabase 
to perform this query. In particular, the text query input filters through the description of each equity in the databse
to filter the equity stocks which include the keyword. The sector input filters this by type of sector. To see the full 
list of sectors please consult (https://github.com/JerBouma/FinanceDatabase/tree/master/Database/Equities/Sectors). The 
default country selected is United States.

The reason why US stocks are selected is because the latter part of the script takes the ticker symbols extracted to make 
a query against (https://github.com/jadchaar/sec-edgar-downloader) to get the latest 10-K reports filed by each company.

From this the script converts the extracted .htm files into text before then applying tokenization techniques to extract 
only the most important keywords of the text (removing stop-words, unnecessary words, entities, etc.).

Finally, the script builds an Latent Dirichlet Allocation (LDA) model to perform topic modelling - the default selected 
number of topics for this unsupervised model is 10 (after having done an hyper-parameter optimisation investigation 
in a separate notebook). This can be changed in the main.py

After applying the model, the dominant topic is attributed to each ticker in an aggregate dataframe which is then saved 
as a .csv in an output directory.


## installation guide
Create a python environment and run pip install -r requirements.txt to download the required database

To download the required language model, run the following in your terminal:
python -m spacy download en_core_web_lg

For more information on the language models provided by Spacy, see: https://spacy.io/models/en (you may, for example, 
wish to use the smaller model version; if so, please make sure to update the main.py accordingly.)