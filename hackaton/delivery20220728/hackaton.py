import utils.utils
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
# Dao
from dao.connector import Connector
# Warnings
import warnings
warnings.filterwarnings("ignore")

def _test():
    model_name = "bert-base-multilingual-uncased"
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline( model=model, tokenizer=tokenizer)
    sentence = "El perro le ladraba a La Gatita .. .. lol #teamlagatita en las playas de Key Biscayne este Memorial day"
    print(nlp(sentence))

if __name__ == '__main__':
    print('######################')
    print('#####Hackaton NLP#####')
    print('######################')
    # connector = Connector()
    # df = connector.get_data_file('big_query_queries/nlp_query.sql')
    # print(df.head())
    from transformers import pipeline
    translator = pipeline("translation_en_to_de")
    text = "Hello world! Hugging Face is the best NLP tool."
    translation = translator(text)

    print(translation)
    print('Succeed!!')