from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
# from google.colab import userdata   #comment this line when working on local system
import os

class ResponseAgent:
    def __init__(self, llm) -> None:
        self.llm = llm        
        self.memory = ConversationBufferMemory(input_key='query')

    def GetPromptTemplate(self):
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template = """<|system|>
                       
                        You are an expert response generator for travel chatbot, reading the given context you should summarize the brief & to the point response. 
                        If user has not given complete information, ask user to give complete information including the date of travel, to or from the place of travel, etc.
                        Your tone should be polite.
                        If you do not know the answer, say "I am not sure about your query. I will ask our human agents to contact you." 
                        Generate the response in a step-by-step procedure.

                        RULES: 
                        1) While doing any kind of bookings consider asking traveller details like name, number, passport details etc if not provided.
                        2) while giving information about flights always ask for dates if not available before giving any information
                        3) while giving information about hotels always ask for dates if not available before giving any information

                        Context: {context}

                        </|system|>
                       
                        <|user|>{query}</|user|>
                       
                        <|assistant|>"""
        )
        return prompt
   
    def GenerateResponse(self, user_question, context):
        prompt = self.GetPromptTemplate()
        # prompt_formatted_str: str = prompt.format(query=user_question, context=context)
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            output_parser=StrOutputParser()
        )
        
        # Get chat history from memory
        # chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # Include all required variables in the invoke call
        response = chain.predict(query=user_question, context=context)
        # response = chain.invoke({
        #     "query": user_question,
        #     "context": context
        # })
        
        # Since we're using StrOutputParser, response is already a string
        return response
   
    def ClearMemory(self):
        self.memory.clear()
   
if __name__ == "__main__":
   
    
    token = os.getenv('OPENAI_API_KEY')

    # Initialize the model and agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=token)
    chat = ResponseAgent(llm)
    # Example conversation
    questions = [
        "I want to book a trip to manali, let me know all possible options",
        "What about hotels there?",
        "How much would it cost for a week?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = chat.GenerateResponse(question, "Ice land is the best hotel in manali")
        print(f"Response: {response}")