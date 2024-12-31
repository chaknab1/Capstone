from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import json
# from google.colab import userdata


class PerformanceEvaluationAgent:

    def __init__(self, json_path, llm):
        self.json_path = json_path
        self.llm = llm

    def RateFromJson(self):
        
        try:
            with open(self.json_path, 'r') as f:
                conversation_data = json.load(f)

            rating_template = """
            1) You are an AI tasked with evaluating the overall satisfaction of a conversation between a human and an agent.
            2) Rate the entire conversation on a scale of 1 to 10 (1: lowest, 10: highest) considering the final_response_feedback, sentiment_feedback, intent_feedback, topic_feedback, current_response_feedback provided in the conversation
            3) Down rate if there are any negative feedback, Do not consider - feedback while rating.
            3) Identify any training needs for the agent and list them. Recognize exceptional service when provided.
            4) Summarize your evaluation in the following format:

            **Overall Satisfaction Score:** (Score between 1 and 10)
            **Feedback:** (Reason for the rating)
            **Training Needs:** (List any training needs for the agent)
            **Exceptional Service:** (Yes/No)

            Conversation:
            {conversation}
            """

            prompt = PromptTemplate.from_template(rating_template)

            # Create an LLMChain using the prompt and model
            chain = prompt | self.llm

            # Pass the updated conversation dictionary to the invoke function
            response = chain.invoke({'conversation': conversation_data})

            # Access the 'content' attribute of the AIMessage object
            response_content = response.content
            print(response_content)
            first_line = response_content.split('\n')[0]
            return first_line
        except Exception as e:
                print(f"Error in analyze_question: {str(e)}")