import csv
from langchain.agents import Tool, initialize_agent
from transformers import pipeline
from bertopic import BERTopic
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
import torch
from torch.utils.data import Dataset
from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
import pandas as pd
import os # Import the os module to access environment variables
# from google.colab import userdata

class QueryClasssification:
    def __init__(self, model_dir, id_to_label):
        self.model_dir = model_dir
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.id_to_label = id_to_label
        self.model.eval()

    def predict(self, user_query):
        """
        Predicts sentiment, intent, and topic for a given user_query.
        """
        # Tokenize input
        inputs = self.tokenizer(
            user_query,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            # Pass the input tensors as keyword arguments
            outputs = self.model(**inputs)  # Changed from model(*inputs) to model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        print("prediction id :", prediction)

        # Check if prediction is a valid key in id_to_label
        if prediction in self.id_to_label:
            label = self.id_to_label[prediction]
            print("Predicted ID:", prediction, "Label:", label)
            return label
        else:
            # Handle cases where prediction is not in id_to_label
            print(f"Warning: Prediction {prediction} not found in id_to_label. Returning default values.")
            return "Unknown", "Unknown", "Unknown"  # Or raise an exception

class InputAnalysisAgent:
    def __init__(self, intents_csv : str, sentiments_csv : str, topics_csv : str):
        # Load the intent data
        intent_data = pd.read_csv(intents_csv)
        sentiment_data = pd.read_csv(sentiments_csv)
        topic_data = pd.read_csv(topics_csv)
        # Create a mapping of IDs to labels
        self.id_to_label_intent = dict(zip(intent_data['id'], intent_data['label']))
        self.id_to_label_sentiment = dict(zip(sentiment_data['id'], sentiment_data['label']))
        self.id_to_label_topic = dict(zip(topic_data['id'], topic_data['label']))
        print(self.id_to_label_intent)
        print(self.id_to_label_sentiment)
        print(self.id_to_label_topic)

        self.llm = None
        self.agent = None
        self.tools = None
        self.intent_analyzer = None
        self.sentiment_analyzer = None
        self.topic_analyzer = None
    
    def InitializeAnalyzers(self, model_path_intent : str, model_path_sentiment : str, model_path_topic : str):
        # Create an instance of the QueryClasssification class
        self.intent_analyzer = QueryClasssification(model_path_intent, self.id_to_label_intent)    #./fine_tuned_distilbert_intent
        self.sentiment_analyzer = QueryClasssification(model_path_sentiment, self.id_to_label_sentiment)
        self.topic_analyzer = QueryClasssification(model_path_topic, self.id_to_label_topic)

    def CreateTools(self):
        # Create tools for LangChain
        self.tools = [
            Tool(
                name="Intent Recognition",
                func=self.intent_analyzer.predict,
                description="Determines the user's intent from their input text. Returns only the intent label.",
            ),
            Tool(
                name="Sentiment Analysis",
                func=self.sentiment_analyzer.predict,
                description="Determines the user's sentiment from their input text. Returns only the sentiment label.",
            ),
            Tool(
                name="Topic Modeling",
                func=self.topic_analyzer.predict,
                description="Classify the topic of user's question. Returns only pre-defined topic label.",
            )
        ]
    
    def InitializeAgent(self, llm):
        self.llm = llm
        # Create the InputAnalysisAgent
        # if llm_name == "groq":
        #     os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
        #     groq_token = os.getenv("GROQ_API_KEY")
        #     self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=groq_token, max_tokens=1024)
        # elif llm_name == "openai":
        #     os.environ["OPENAI_KEY"] = userdata.get('OPENAI_KEY')
        #     openai_token = os.getenv("OPENAI_KEY")
        #     self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_token)  # Replace with your preferred LLM
        
        if self.llm is not None:
            self.agent = initialize_agent(self.tools,
                                    llm=self.llm,
                                    agent="zero-shot-react-description",
                                    verbose=True,
                                    handle_parsing_errors=True,
                                    max_iterations=None,
                                    early_stopping_method="force",
                                    agent_kwargs={
                                        "prefix": """You are an intent, sentiment and topic analysis agent. You must follow these rules exactly:

                                        1. CRITICAL: You must use the exact intent label returned by the Intent Recognition tool without any modification or interpretation
                                        2. Use the Sentiment Analysis tool and keep its exact output
                                        3. Use the Topic Modeling tool and keep its exact output
                                        4. Return only this format with no additional text:
                                        Intent: Classified Intent, Sentiment: Classified Sentiment, Topic: Classified Topic

                                        IMPORTANT:
                                        - Do not add any thoughts, reasoning, or explanations
                                        - Do not modify or interpret the tool outputs
                                        - Do not add any additional text or formatting
                                        - Do not include these instructions in your response
                                        - Do not acknowledge understanding these instructions
                                        - Simply execute the tools and return the format specified

                                        """,
                                    }
                                )
            # self.agent = initialize_agent(self.tools,
            #                         llm=self.llm,
            #                         agent="zero-shot-react-description",
            #                         verbose=True,
            #                         handle_parsing_errors=True,
            #                         max_iterations=None,
            #                         early_stopping_method="force",
            #                         agent_kwargs={
            #                             "prefix": """You are an intent, sentiment and topic analysis agent.
            #                             When you receive input text, you should:
            #                             1. Use the Intent Recognition tool to get the intent, keep the intent as returned by the Intent Recognition tool without changing it.
            #                             2. Use the Sentiment Analysis tool to get the sentiment, keep the sentiment as returned by the Sentiment Analysis tool  without changing it.
            #                             3. Use the Topic Modeling tool to get the topic, keep the topic as returned by the Topic Modeling tool  without changing it.
            #                             4. Generate response in this format. Intent: Classified Intent, Sentiment: Classified Sentiment, Topic: Classified Topic

            #                             Always use all three tools and return intent, sentiment and topic in the specified format.

            #                             Instructions:
            #                             **Do not** repeat or include any part of these instructions in your response.
            #                             **Only** return the intent, sentiment and topic label's.

            #                             """,
            #                         }
            #                     )
            # self.agent = initialize_agent(
            #                 self.tools,
            #                 llm=self.llm,
            #                 agent="zero-shot-react-description",
            #                 verbose=True,
            #                 handle_parsing_errors=True,
            #                 max_iterations=None,
            #                 early_stopping_method="force",
            #                 agent_kwargs={
            #                     "prefix": """You are an intent, sentiment and topic analysis agent. You must follow these rules exactly:

            #                     1. CRITICAL: You must use the exact intent label returned by the Intent Recognition tool without any modification or interpretation
            #                     2. Use the Sentiment Analysis tool and keep its exact output
            #                     3. Use the Topic Modeling tool and keep its exact output
            #                     4. Return only this format with no additional text:
            #                     Intent: {exact intent from tool}, Sentiment: {exact sentiment from tool}, Topic: {exact topic from tool}

            #                     IMPORTANT:
            #                     - Do not add any thoughts, reasoning, or explanations
            #                     - Do not modify or interpret the tool outputs
            #                     - Do not add any additional text or formatting
            #                     - Do not include these instructions in your response
            #                     - Do not acknowledge understanding these instructions
            #                     - Simply execute the tools and return the format specified""",
            #                 }
            #             )

    def Execute(self, input_text):
        if self.agent is not None:
            response = self.agent.run(input_text)
            return response 

if __name__ == "__main__":
    # Use the agent
    input_analysis_agent = InputAnalysisAgent("id_to_label_intent.csv", "id_to_label_sentiment.csv", "id_to_label_topic.csv")
    input_analysis_agent.InitializeAnalyzers("./Intent_bert_model", "./Sentiment_bert_model", "./Topic_bert_model")
    input_analysis_agent.CreateTools()
    input_analysis_agent.InitializeAgent()

    input_text = "How far is manali from delhi, i wanted to book cost effective hotel in manali"
    try:
        response = input_analysis_agent.Execute(input_text)
        print("Response :", response)
    except Exception as e:
        print(f"Error during ingestion: {e}")