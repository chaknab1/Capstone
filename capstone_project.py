from amadeus import Client, ResponseError
import os
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
import gradio as gr
import time
import sys
import json
from datetime import datetime

# sys.path.append('E:/Personal/IISC/Capstone/IISC_Capstone_Project/')
from input_analysis_agent import InputAnalysisAgent
from response_agent import ResponseAgent
from data_retrival_agent import TravelPlanningAgent
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from performance_evaluation_agent import PerformanceEvaluationAgent

custom_css = """
.travel-chatbot-heading {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
    background: #4169E1;  /* Royal Blue */
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    font-family: 'Arial', sans-serif;
    font-size: 36px;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""

class LLMQuestionRefiner:
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.refine_template = ChatPromptTemplate.from_messages([
            ("system", """You are a LLMQuestionRefiner as a user assistance that refines my input by adding relevant context 
            from previous conversation. Create a single, clear, and concise input that includes important 
            context from the chat history. Do not add explanations or additional text.
            
            Rules:
            1. Only output the refined question, nothing else
            2. Keep the question natural and straight forward
            3. Only add context that is directly relevant
            4. If no relevant context exists, return the original question unchanged
            5. Don't make assumptions about context that isn't in the history
            
            Example 1:
            History: "I'm planning a trip to Paris next month"
            Question: "What museums should I visit?"
            Output: "What museums should I visit in Paris?"
            
            Example 2:
            History: ""
            Question: "I am planning a trip to Mumbai"
            Output: "I am planning a trip to Mumbai"
             
            Example 2:
            History: "I love India"
            Question: "I am planning a trip to Mumbai"
            Output: "I am planning a trip to Mumbai"
            """),
            ("human", """Previous conversation:
            {chat_history}
            
            Current question: {question}""")
        ])
        
    def add_message(self, message: str, is_user: bool = True):
        """Add a message to conversation history"""
        if is_user:
            self.memory.chat_memory.add_user_message(message)
        else:
            self.memory.chat_memory.add_ai_message(message)
        
    def refine_question(self, question: str) -> str:
        """Refine the question using LLM and conversation context"""
        history = self.memory.load_memory_variables({})["chat_history"]
        chain = self.refine_template | self.llm
        result = chain.invoke({
            "chat_history": history,
            "question": question
        })
        return result.content
    
    def clear_memory(self):
        """Clear all conversation history"""
        self.memory.clear()


global_final_response_feedback = "-"
global_sentiment_feedback = "-"
global_intent_feedback = "-"
global_topic_feedback = "-"
global_current_response_feedback = "-"

# Global variables for storing previous conversation
global_prev_question = "empty"
global_prev_sentiment = "empty"
global_prev_intent = "empty"
global_prev_topic = "empty"
global_prev_answer_str = "empty"
global_prev_response_time = "empty"


if __name__ == "__main__":
    amadeus = Client(
        client_id="QBSBUsh1p6DwLRtF2KpDavFECGYvZ4Ui",
        client_secret="fL4XnNYLFeitqO4I",
    )

    token_groq = os.getenv("GROQ_API_KEY")
    llm_groq = ChatGroq(
        temperature=0.1,
        model_name="mixtral-8x7b-32768",
        groq_api_key=token_groq,
        max_tokens=1024
    )

    token = os.environ.get("OPENAI_KEY")
    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo", openai_api_key=token)

    response_agent = ResponseAgent(llm)

    input_analysis_agent = InputAnalysisAgent("./id_to_label_intent.csv", "./id_to_label_sentiment.csv", "./id_to_label_topic.csv")
    input_analysis_agent.InitializeAnalyzers("./models/fine_tuned_distilbert_intent", "./models/fine_tuned_distilbert_sentiment", "./models/fine_tuned_distilbert_topic")
    input_analysis_agent.CreateTools()
    input_analysis_agent.InitializeAgent(llm)

    data_retrival_agent = TravelPlanningAgent(llm, amadeus)

    performance_evaluation_agent = PerformanceEvaluationAgent("./conversation_history.json", llm_groq)

    refiner = LLMQuestionRefiner(llm_groq)

    # questions = ["I am planning a trip to USA",
    #             "List few places to visit",
    #             "help me find a hotel",
    #             "help me book flight from ahmedabad to USA",
    #             "which one is the cheapest option"]
    
    # for question in questions:
    #     refiner.add_message(question, True)
    #     refined_question = refiner.refine_question(question)
    #     # refined_question = question
    #     print("ORIGINAL QUESTION: ", question)
    #     print("REFINED QUESTION: ", refined_question)

    memory = ConversationBufferMemory(input_key='query')

    question = "Plan a trip to Mumbai"
    response = input_analysis_agent.Execute(question)
    print(response)
    intent = response.split("Intent: ")[1].split(",")[0]  # Gets "booking_assistance"
    intent = intent.replace("_", " ").title()
    intent = intent.replace("\\", "").title()

    sentiment = response.split("Sentiment: ")[1].split(",")[0]  # Gets "positive"
    sentiment = sentiment.replace("_", " ").title()
    sentiment = sentiment.replace("\\", "").title()

    topic = response.split("Topic: ")[1]  # Gets "lodging"
    topic = topic.replace("_", " ").title()
    topic = topic.replace("\\", "").title()

    # Placeholder analysis - replace with actual implementation
    print(intent, sentiment, topic)

    context = data_retrival_agent.run(intent, question)
    print(context)

    answer = response_agent.GenerateResponse(question, context)
    print("FINAL RESPONSE:", answer)


    def analyze_question(question, history):
        """
        Analyzes the user question and returns relevant information
        """
        try:
            global global_prev_question, global_prev_sentiment, global_prev_intent
            global global_prev_topic, global_prev_answer_str, global_prev_response_time

            global global_final_response_feedback, global_sentiment_feedback, global_intent_feedback, global_topic_feedback, global_current_response_feedback

            # Save previous conversation if exists
            if global_prev_question != "empty":
                save_to_json(
                    global_prev_question,
                    global_prev_sentiment,
                    global_prev_intent,
                    global_prev_topic,
                    global_prev_answer_str,
                    global_prev_response_time
                )

            # final_rating_output = performance_evaluation_agent.RateFromJson()

            start_time = time.time()

            refiner.add_message(question, True)
            refined_question = refiner.refine_question(question)
            print("\n\n REFINED QUESTION: ", refined_question)

            # Get the analysis results
            response = input_analysis_agent.Execute(refined_question)
            intent = response.split("Intent: ")[1].split(",")[0]
            intent = intent.replace("_", " ").replace("\\", "").title()

            sentiment = response.split("Sentiment: ")[1].split(",")[0]
            sentiment = sentiment.replace("_", " ").replace("\\", "").title()

            topic = response.split("Topic: ")[1]
            topic = topic.replace("_", " ").replace("\\", "").title()

            # Get context and generate response

            context = data_retrival_agent.run(intent, refined_question)
            print("\n\n CONTEXT: ", context)
            answer = response_agent.GenerateResponse(refined_question, context)

            # refiner.add_message(answer, False)

            # Calculate response time
            response_time = int((time.time() - start_time) * 1000)


            # Format history properly for Gradio Chatbot
            history = history or []
            # Ensure the answer is a string
            answer_str = str(answer)
            history.append([question, answer_str])

            global_final_response_feedback = "-"
            global_sentiment_feedback = "-"
            global_intent_feedback = "-"
            global_topic_feedback = "-"
            global_current_response_feedback = "-"

            # Store current conversation for next save
            global_prev_question = question
            global_prev_sentiment = sentiment
            global_prev_intent = intent
            global_prev_topic = topic
            global_prev_answer_str = answer_str
            global_prev_response_time = response_time

            return (
                "",
                history,      # Properly formatted history
                sentiment,
                intent,
                topic,
                answer_str,   # Ensure we return the string version
                response_time,
                refined_question,
                final_rating_output
            )
        except Exception as e:
            print(f"Error in analyze_question: {str(e)}")

            # Return empty/error values in case of failure
            return (
                "",
                history or [],
                "Error",
                "Error",
                "Error",
                f"An error occurred: {str(e)}",
                0,
                "Error",
                "Error"
            )

    def clear_inputs(questions_chain):
        """
        Clears all input and output fields and stores last feedback in json
        """
        global global_prev_question, global_prev_sentiment, global_prev_intent
        global global_prev_topic, global_prev_answer_str, global_prev_response_time

        # Save previous conversation if exists
        if global_prev_question != "empty":
            save_to_json(
                global_prev_question,
                global_prev_sentiment,
                global_prev_intent,
                global_prev_topic,
                global_prev_answer_str,
                global_prev_response_time
            )
        # refiner.clear_memory()
        # memory.clear()
        questions_chain = ""
        return (
            "",          # Clear question input
            [],          # Clear chat history - changed from None to []
            "",          # Clear sentiment
            "",          # Clear intent
            "",          # Clear topic
            "",          # Clear answer
            "",          # Clear response time
            "",          # Clear final rating
            ""
        )

    def end_conversation(questions_chain):
        """
        Clears all input and output fields and stores last feedback in json and display final rating
        """
        global global_prev_question, global_prev_sentiment, global_prev_intent
        global global_prev_topic, global_prev_answer_str, global_prev_response_time

        # Save previous conversation if exists
        if global_prev_question != "empty":
            save_to_json(
                global_prev_question,
                global_prev_sentiment,
                global_prev_intent,
                global_prev_topic,
                global_prev_answer_str,
                global_prev_response_time
            )
        questions_chain = ""
        final_rating_output = performance_evaluation_agent.RateFromJson()
        return (
            "",          # Clear question input
            "",             #Clear refined question
            [],          # Clear chat history - changed from None to []
            "",          # Clear sentiment
            "",          # Clear intent
            "",          # Clear topic
            "",          # Clear answer
            "",          # Clear response time
            final_rating_output
        )

    def capture_feedback_final_response(component_type, feedback_value):
        """
        Captures and processes user feedback
        """
        global global_final_response_feedback
        global_final_response_feedback = feedback_value
        print(f"Received {component_type} feedback: {feedback_value}")
        return f"Thank you for your {feedback_value} feedback on {component_type}!"

    def capture_feedback_sentiment(component_type, feedback_value):
        """
        Captures and processes user feedback
        """
        global global_sentiment_feedback
        global_sentiment_feedback = feedback_value
        print(f"Received {component_type} feedback: {feedback_value}")
        return f"Thank you for your {feedback_value} feedback on {component_type}!"

    def capture_feedback_intent(component_type, feedback_value):
        """
        Captures and processes user feedback
        """
        global global_intent_feedback
        global_intent_feedback = feedback_value
        print(f"Received {component_type} feedback: {feedback_value}")
        return f"Thank you for your {feedback_value} feedback on {component_type}!"

    def capture_feedback_topic(component_type, feedback_value):
        """
        Captures and processes user feedback
        """
        global global_topic_feedback
        global_topic_feedback = feedback_value
        print(f"Received {component_type} feedback: {feedback_value}")
        return f"Thank you for your {feedback_value} feedback on {component_type}!"

    def capture_feedback_current_response(component_type, feedback_value):
        """
        Captures and processes user feedback
        """
        global global_current_response_feedback
        global_current_response_feedback = feedback_value
        print(f"Received {component_type} feedback: {feedback_value}")
        return f"Thank you for your {feedback_value} feedback on {component_type}!"

    def save_to_json(question, sentiment, intent, topic, answer, response_time):
        global global_final_response_feedback, global_sentiment_feedback, global_intent_feedback, global_topic_feedback, global_current_response_feedback

        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_data": {
                "question": question,
                "sentiment": sentiment,
                "intent": intent,
                "topic": topic,
                "answer": answer,
                "response_time": response_time,
                "final_response_feedback": global_final_response_feedback,
                "sentiment_feedback": global_sentiment_feedback,
                "intent_feedback": global_intent_feedback,
                "topic_feedback": global_topic_feedback,
                "current_response_feedback": global_current_response_feedback,
            },
        }

        try:
            with open('./conversation_history.json', 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        existing_data.append(data)

        with open('./conversation_history.json', 'w') as f:
            json.dump(existing_data, f, indent=4)


    # Create the Gradio interface
    with gr.Blocks(css=custom_css) as demo:
        # Artistic Heading
        # gr.HTML(
        #     """
        #     <div class="travel-chatbot-heading">
        #         ✈️ RoamRight : Your AI Travel Companion 🌎
        #     </div>
        #     """
        # )
        gr.HTML(
            """
            <div class="travel-chatbot-container">
                <div class="travel-chatbot-heading">
                    ✈️ RoamRight : Your AI Travel Companion 🌎
                </div>
                <div class="travel-chatbot-subheading">
                    Got questions about boarding, lodging, or travel? Let’s journey together—don’t forget to rate us!
                </div>
                <style>
                    .travel-chatbot-container {
                        text-align: center;
                        padding: 20px 0;
                    }
                    .travel-chatbot-heading {
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .travel-chatbot-subheading {
                        font-size: 18px;
                        color: #666;
                        font-style: bold;
                    }
                </style>
            </div>
            """
        )

        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask your travel-related question here..."
                )
                refined_question = gr.Textbox(
                    label="Refined Question",
                    interactive=False
                )
                chatbot = gr.Chatbot(
                    label="Conversation History",
                    height=300
                )
                with gr.Row():
                    chat_thumbs_up = gr.Button("👍")
                    chat_thumbs_down = gr.Button("👎")

                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.Button("Clear")

                final_rating_output = gr.Textbox(
                    label="Auto Performance Evaluation",
                    interactive=False
                )

                with gr.Row():
                    end_conversation_btn = gr.Button("End Conversation", scale=2)

            # Right Panel
            with gr.Column(scale=1):
                # Response Time Display
                response_time_output = gr.Textbox(
                    label="Response Time (ms)",
                    interactive=False
                )

                # Sentiment Row
                sentiment_output = gr.Textbox(
                    label="Sentiment",
                    interactive=False
                )
                with gr.Row():
                    sentiment_up = gr.Button("👍")
                    sentiment_down = gr.Button("👎")

                # Intent Row
                intent_output = gr.Textbox(
                    label="Intent",
                    interactive=False
                )
                with gr.Row():
                    intent_up = gr.Button("👍")
                    intent_down = gr.Button("👎")

                # Topic Row
                topic_output = gr.Textbox(
                    label="Topic",
                    interactive=False
                )
                with gr.Row():
                    topic_up = gr.Button("👍")
                    topic_down = gr.Button("👎")

                # Answer Row
                answer_output = gr.Textbox(
                    label="Answer",
                    interactive=False
                )
                with gr.Row():
                    answer_up = gr.Button("👍")
                    answer_down = gr.Button("👎")

        # Event handlers
        submit_btn.click(
            analyze_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot, sentiment_output, intent_output, topic_output, answer_output, response_time_output, refined_question, final_rating_output]
        )

        question_input.submit(
            analyze_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot, sentiment_output, intent_output, topic_output, answer_output, response_time_output, refined_question, final_rating_output]
        )

        clear_btn.click(
            clear_inputs,
            outputs=[question_input, chatbot, sentiment_output, intent_output, topic_output, answer_output, response_time_output, final_rating_output, refined_question]
        )

        end_conversation_btn.click(
            end_conversation,
            outputs=[question_input, refined_question, chatbot, sentiment_output, intent_output, topic_output, answer_output,response_time_output, final_rating_output]
        )

        # Left panel feedback handlers
        chat_thumbs_up.click(
            lambda: capture_feedback_final_response("chat", "positive"),
            outputs=gr.Textbox(visible=False)
        )
        chat_thumbs_down.click(
            lambda: capture_feedback_final_response("chat", "negative"),
            outputs=gr.Textbox(visible=False)
        )

        # Right panel feedback handlers
        # Sentiment feedback
        sentiment_up.click(
            lambda: capture_feedback_sentiment("sentiment", "positive"),
            outputs=gr.Textbox(visible=False)
        )
        sentiment_down.click(
            lambda: capture_feedback_sentiment("sentiment", "negative"),
            outputs=gr.Textbox(visible=False)
        )

        # Intent feedback
        intent_up.click(
            lambda: capture_feedback_intent("intent", "positive"),
            outputs=gr.Textbox(visible=False)
        )
        intent_down.click(
            lambda: capture_feedback_intent("intent", "negative"),
            outputs=gr.Textbox(visible=False)
        )

        # Topic feedback
        topic_up.click(
            lambda: capture_feedback_topic("topic", "positive"),
            outputs=gr.Textbox(visible=False)
        )
        topic_down.click(
            lambda: capture_feedback_topic("topic", "negative"),
            outputs=gr.Textbox(visible=False)
        )

        # Answer feedback
        answer_up.click(
            lambda: capture_feedback_current_response("answer", "positive"),
            outputs=gr.Textbox(visible=False)
        )
        answer_down.click(
            lambda: capture_feedback_current_response("answer", "negative"),
            outputs=gr.Textbox(visible=False)
        )

    demo.launch(share=True)