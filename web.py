import requests
import numpy as np
import streamlit as st
import tensorflow as tf
import tweepy
from transformers import pipeline , TFAutoModelForSequenceClassification, AutoTokenizer




APIKey = "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"
CONSUMER_KEY = "eJL1xOgPnXVx0DzCr5pGa8lNv"
CONSUMER_SECRET = "iDBuPdCEXZQDzsRqvNtkVcIhcdvlT8x8aW74VTm1EqXcIaPmrZ"
OAUTH_TOKEN = "1420293020080082948-Qo8PBaf5oXA1xrPryabo3C3g09xdBf"
OAUTH_TOKEN_SECRET = "HzW0KllX3NRls7pOKA0cPIbNEyAFWON9wgVODcRwlrVBi"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAABKRewEAAAAAem8kRGNx2U5tGTmD%2BtikqmENETI%3DCqzEWLXGEmM8WKNXRRnW0Tke4QlWw2sihgwtjpVYwnAR0QD6bo"
twitterAPI = tweepy.Client(bearer_token = BEARER_TOKEN)



def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

def get_tweet(link) :
    twt_id = (link.split('/')[-1])
    twt_id = requests.sub(r'\?.+', '', twt_id)
    tweetContent = twitterAPI.get_tweet(twt_id)
    tweetContent = str(tweetContent[0]).split()
    return (" ".join(tweetContent))

def prediction(news) :
    
#     test_data = word_tokenize(news)
#     v1 = d2v_model.infer_vector(test_data)
#     similar_doc = d2v_model.docvecs.most_similar(positive=[v1])
#     if similar_doc[0][1]>=0.725 :

    sentences=[news]
    tokenized = cae_model_tokenizer(sentences, return_tensors="np", padding="longest")
    outputs = classification_after_embedding_model(tokenized).logits
    classifications = np.argmax(outputs, axis=1)
    if classifications[0]==0 :
        textToReply = "The given news needs to be verified"
    else :
        textToReply = "The given news is TRUE"
        
    return textToReply

def detect_news(update, context) :
    news = update.message.text

    update.message.reply_text("Waiting for the output....")

    if news[:5]=='https' :
        news = get_tweet(news)
    textToReply = prediction(news)
    finalNewsFeatures = getNewsFeatures(news)
#     finalWikipediaResults = topNBert(news, topNSimilar(news, content(news, keywords(news))))
#     kws=keywords(news)
#     cnt=content(news,kws)
#     tns=topNSimilar(news,cnt)
#     tnb=topNBert(news,tns)
    
    update.message.reply_text(textToReply)
    update.message.reply_text("The news features are : ")
    for key , val in finalNewsFeatures.items() :
        update.message.reply_text(key + " : " + val)
#     update.message.reply_text("We found these related articles on the web :")
#     for i in tnb :
#         update.message.reply_text(i[0])

def load_models() :
     global clickBaitModel, classification_after_embedding_model, sentimentModel, biasModel, cae_model_tokenizer
     clickBaitModel = pipeline(model="elozano/bert-base-cased-clickbait-news", tokenizer="elozano/bert-base-cased-clickbait-news")
     classification_after_embedding_model = TFAutoModelForSequenceClassification.from_pretrained('pururaj/Test_model')
     sentimentModel = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")
     biasModel = pipeline(model="d4data/bias-detection-model", tokenizer="d4data/bias-detection-model")
     cae_model_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


def getNewsFeatures(inputText) :
    finalNewsFeatures = {}
    results_model1 = clickBaitModel(inputText)[0]
    if(results_model1['label']=='Clickbait') :
        finalNewsFeatures.__setitem__('Clickbait probability', str(round((results_model1['score']*100), 2))+"%")
    else :
        finalNewsFeatures.__setitem__('Clickbait probability', str(round(((1-results_model1['score'])*100), 2))+"%")
    results_model2 = sentimentModel(inputText)[0]
    finalNewsFeatures.__setitem__('Sentiment', results_model2['label'])
    results_model3 = biasModel(inputText)[0]
    if(results_model3['label']=='Biased') :
        finalNewsFeatures.__setitem__('Biased percentage', str(round((results_model3['score']*100), 2))+"%")
    else :
        finalNewsFeatures.__setitem__('Biased percentage', str(round(((1-results_model3['score'])*100), 2))+"%")
    results_model4 = Detoxify('original').predict(inputText)
    finalNewsFeatures.__setitem__('Toxicity percentage', str(round((results_model4['toxicity']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Obscene percentage', str(round((results_model4['obscene']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Insult percentage', str(round((results_model4['insult']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Hatred percentage', str(round((results_model4['identity_attack']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Threat percentage', str(round((results_model4['threat']*100), 2))+"%")
    return finalNewsFeatures


def main():
    st.title("Fake News Detection App")
    st.write("Enter a news article or tweet to check if it's fake or not.")

    user_input = st.text_area("Enter the news or tweet", height=200)
    if st.button("Check"):
        # Load the model
        model = load_model()

        # Perform prediction using the loaded model
        prediction = detect_news(user_input, model)

        # Display the prediction result
        if prediction == 0:
            st.write("The news is FAKE.")
        else:
            st.write("The news is TRUE.")

        # Display news features
        st.subheader("News Features")
        news_features = getNewsFeatures(user_input)
        for key, value in news_features.items():
            st.write(f"- {key}: {value}")

if __name__ == "__main__":
    main()
