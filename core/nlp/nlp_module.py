# File: nlp/nlp_module.py
# מודול עיבוד שפה טבעית עם NLTK

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# הורדת משאבים אם זו הפעם הראשונה
nltk.download('punkt')
nltk.download('stopwords')

def process_text(text):
    """ מפרק טקסט למילים ומסנן מילים מיותרות (Stopwords). """
    tokens = word_tokenize(text)  # פירוק משפט למילים
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    return filtered_tokens

if __name__ == '__main__':
    sample_text = "Hello, this is an AI project using NLP!"
    tokens = process_text(sample_text)
    print("Processed tokens:", tokens)
