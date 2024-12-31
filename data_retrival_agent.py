import os
import json
# from google.colab import userdata
import requests

from amadeus import Client, ResponseError

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentType
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq

# --- Amadeus API Setup ---
# amadeus = Client(
#     client_id="QBSBUsh1p6DwLRtF2KpDavFECGYvZ4Ui",
#     client_secret="fL4XnNYLFeitqO4I",
# )
# # os.environ["OPENAI_KEY"] = userdata.get('OPENAI_KEY')
# # openai.api_key = os.environ.get("OPENAI_KEY")
# # llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo", openai_api_key=openai.api_key)

# os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY') #comment this line when working on local system
# groq_token = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(
#             temperature=0.1,
#             model_name="mixtral-8x7b-32768",
#             groq_api_key=groq_token,
#             max_tokens=1024
#         )

# --- Agent Classes ---

def json_to_plain_text(data, indent=0):
    plain_text = ""
    if isinstance(data, dict):
        for key, value in data.items():
            plain_text += " " * indent + f"{key}: "
            if isinstance(value, (dict, list)):
                plain_text += "\n" + json_to_plain_text(value, indent + 2)
            else:
                plain_text += f"{value}\n"
    elif isinstance(data, list):
        for index, item in enumerate(data, start=1):
            plain_text += " " * indent + f"{index}. "
            if isinstance(item, (dict, list)):
                plain_text += "\n" + json_to_plain_text(item, indent + 2)
            else:
                plain_text += f"{item}\n"
    else:
        plain_text += " " * indent + f"{data}\n"
    return plain_text

class getIATACode:
   def __init__(self):
      pass
   def run(self, city, amadeus):
      try:
          # Add parameter validation
          if not city or not isinstance(city, str):
              raise ValueError("City must be a non-empty string")
              
          if not amadeus:
              raise ValueError("Amadeus client instance is required")
              
          # Make the API call with more specific parameters
          response = amadeus.reference_data.locations.get(
              keyword=city,
              subType=amadeus.reference_data.locations.LocationType.CITY,
              include='CITY'
          )
          
          # Check if we got any results
          if response.data:
              return response.data[0]['iataCode']
          else:
              return None
              
      except ResponseError as error:
          # Handle specific Amadeus API errors
          error_msg = f"Amadeus API Error: {error.code} - {error.description}"
          print(error_msg)
          return None
          
      except Exception as e:
          # Handle unexpected errors
          print(f"Unexpected error: {str(e)}")
          return None
      # try:
      #   response = amadeus.reference_data.locations.get(keyword=city, subType='CITY').data[0]
      #   return response['iataCode']
      # except Exception as e:
      #   print(f"Error during ingestion: {e}")
      # # except ResponseError as error:
      # #   return f"Amadeus API Error: {error}"

class getGeoCode:
   def __init__(self):
      pass
   def run(self, city, amadeus):
      try:
          response = amadeus.reference_data.locations.get(keyword=city, subType='CITY')
          if response.data:  # Check if the list is not empty
              return response.data[0]['geoCode']
          else:
              print(f"No results found for city: {city}")
              return None
      except Exception as e:
          print(f"Error during ingestion: {e}")

class FlightAgent:
    def __init__(self):
      pass

    def run(self, prompt,intent, llm, amadeus):
      try:
        template = """1. Analyze the  user input: {prompt} to understand the intent of the user
        If the user wants to book a flight only say 'book_flight'.
        If the user wants to check the current status of a flight only say 'flight_status''"""
        prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        result = llm_chain.run(prompt)
        if "book_flight" in result:
          print("Book Flight")
          return FlightSearchAgent().run(prompt, llm, amadeus)
        elif "flight_status" in result:
          print("Flight Status")
          return FlightScheduleAgent().run(prompt, llm, amadeus)
        else:
          print("Other :", result)
          return GetTravelInfo().run(prompt,intent, llm, amadeus)
      except Exception as e:
        return GetTravelInfo().run(prompt,intent, llm, amadeus)

class FlightSearchAgent:
    def __init__(self):
      pass

    def run(self, prompt, intent, llm, amadeus):
      
      try:
        template = """1. Analyze the  user input: {prompt} and Extract the user specifications for origin city/country, destination city/country, departure date, return date, number of adults and children
        2. If any information is missing, assume 1 adult and 0 children. 3. Also, assume today's date as the departure date if not specified. 4. Return dates in yyyy-mm-dd format 5. If year is missing assume the year is 2025
        5. Return the result in JSON format with the following structure: {{"origin": "origin_city", "destination": "destination_city", "departure_date": "departure_date", "return_date": "return_date", "adults": "number_of_adults", "children": "number_of_children"}}"""
        prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        extraction_result_str = llm_chain.run(prompt)

        try:
          extraction_result = json.loads(extraction_result_str)
          if 'return_date' not in extraction_result:
            extraction_result['return_date'] = None
          try:
            extraction_result['origin'] = getIATACode().run(extraction_result['origin'], amadeus)
            extraction_result['destination'] = getIATACode().run(extraction_result['destination'], amadeus)
          except Exception as e:
            return GetTravelInfo().run(prompt, intent, llm, amadeus)
          
          response = amadeus.shopping.flight_offers_search.get(
              originLocationCode=extraction_result['origin'],
              destinationLocationCode=extraction_result['destination'],
              departureDate=extraction_result['departure_date'],
              returnDate=extraction_result['return_date'],
              adults=extraction_result['adults'],
          )
          # Handle API errors
          # Store API response in a dictionary
          result_dict = {"flight_search_results": response.data}
          # Convert dictionary to JSON string
          result_json = json.dumps(result_dict, indent=4)
          formatted_response = json_to_plain_text(result_json, 0)# ... (Process data as needed, e.g., extract names, ratings, etc.) ...

          return result_json
          # Display JSON data
          # return result_json
        except ResponseError as error:
          return f"Amadeus API Error: {error}"
        except Exception as e:
            return GetTravelInfo().run(prompt,intent, llm, amadeus)
        except json.JSONDecodeError:
          return GetTravelInfo().run(prompt,intent, llm, amadeus)
      except Exception as e:
        return GetTravelInfo().run(prompt,intent, llm, amadeus)

class FlightScheduleAgent:
    def __init__(self):
      pass

    def run(self, prompt, intent, llm, amadeus):
      try:
        template = """1. Analyze the  user input: {prompt} and extract the user specifications for flight carrier or airline code, flight number and departure date
        2. Also, assume today's date as the departure date if not specified. 4. Return date in yyyy-mm-dd format 5. If year is missing assume the year is 2025
        5. Return the result in JSON format with the following structure: {{"airline_code": "airline code", "flight_number": "flight number", "departure_date": "departure_date"}}"""
        prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        extraction_result_str = llm_chain.run(prompt)
        import json
        try:
          extraction_result = json.loads(extraction_result_str)
          response = amadeus.schedule.flights.get(
              carrierCode=extraction_result['airline_code'],
              flightNumber=extraction_result['flight_number'],
              scheduledDepartureDate=extraction_result['departure_date']
          )
          # Handle API errors
          # Store API response in a dictionary
          result_dict = {"flight_schedule_data": response.data}
          # Convert dictionary to JSON string
          result_json = json.dumps(result_dict, indent=4)
          formatted_response = json_to_plain_text(result_json, 0)# ... (Process data as needed, e.g., extract names, ratings, etc.) ...

          return result_json
          # Display JSON data
          # return result_json
        except ResponseError as error:
          return f"Amadeus API Error: {error}"
        except Exception as e:
            return f"An error occurred: {e}"
        except json.JSONDecodeError:
          return GetTravelInfo().run(prompt,intent, llm, amadeus)
      except Exception as e:
        return GetTravelInfo().run(prompt,intent, llm, amadeus)

class HotelAgent:
  def __init__(self):
    pass

  def run(self, prompt, intent, llm, amadeus):
    try:
      # Extract hotel details
      hotel_details_prompt_template = """1. Analyze the  user input: {prompt} and extract the user specifications for city, checkin date and checkout date
      2. Also, assume today's date as the checkin date if not specified 3. If checkout date is not specified, Assume one day after the checkin date as the checkout date
      4. Return date in yyyy-mm-dd format 5. If year is missing assume the year is 2025
      6. Return the result in JSON format with the following structure: {{"city": "city","checkin_date": "checkin date","checkout_date": "checkout date"}}
      7. Dont include any text other than the infomration in JSON format"""
      prompt_template = PromptTemplate(input_variables=["prompt"], template=hotel_details_prompt_template)
      hotel_details_chain = LLMChain(llm=llm, prompt=prompt_template)
      hotel_details = hotel_details_chain.run(prompt)
      hotel_details_json = json.loads(hotel_details)
      city = hotel_details_json['city']
      city = getIATACode().run(city, amadeus)
      checkin_date = hotel_details_json['checkin_date']
      checkout_date = hotel_details_json['checkout_date']
      # Call Amadeus hotel search API
      response = amadeus.reference_data.locations.hotels.by_city.get(cityCode=city,radius=20,radiusUnit="km")
      hotel_ids = [hotel['hotelId'] for hotel in response.data[:50]]
      # Handle API errors
      try:
        detailed_response=amadeus.shopping.hotel_offers_search.get(hotelIds=hotel_ids,
                                                                   checkInDate=checkin_date,
                                                                   checkOutDate=checkout_date)
         # Store API response in a dictionary
        result_dict = {"hotel search results": detailed_response.data}
          # Convert dictionary to JSON string
        result_json = json.dumps(result_dict, indent=4)
        formatted_response = json_to_plain_text(result_json, 0)# ... (Process data as needed, e.g., extract names, ratings, etc.) ...

        return result_json
        # return result_json
      except ResponseError as error:
        return f"Amadeus API Error: {error}"
      except Exception as e:
        return f"An error occurred: {e}"
    except ResponseError as error:
      return f"Amadeus API Error: {error}"
    except Exception as e:
        return GetTravelInfo().run(prompt,intent, llm, amadeus)

class PointsOfInterestAgent:
    def __init__(self):
        self.tripadvisor_api_key = 'C37C286FAC804EBF87E8899583D650E1'

    def run(self, prompt, intent, llm, amadeus):
        try:
            category = None
            # 1. Extract city and categories:
            poi_details_prompt_template = """1. Extract city from the following user input: {prompt} and return only city name"""
            prompt_template = PromptTemplate(input_variables=["prompt"], template=poi_details_prompt_template)
            poi_details_chain = LLMChain(llm=llm, prompt=prompt_template)
            poi_details = poi_details_chain.run(prompt)

            # 2. Get latitude and longitude using getGeoCode:
            latitude = getGeoCode().run(poi_details, amadeus)['latitude']
            longitude = getGeoCode().run(poi_details, amadeus)['longitude']

            # 3. Construct the TripAdvisor API request:
            url = "https://api.content.tripadvisor.com/api/v1/location/nearby_search"
            headers = {
                "accept": "application/json",

            }

            params = {
                "latLong": f"{latitude},{longitude}",  # Combine lat and long,
                "key":self.tripadvisor_api_key ,
                "category": category,  # Assuming categories is a comma-separated string
                "radius": 10,  # Adjust as needed
                "radiusUnit":'km'
            }
            # 4. Make the API request:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            # 5. Process the response:
            data = response.json()
            formatted_response = json_to_plain_text(data, 1)# ... (Process data as needed, e.g., extract names, ratings, etc.) ...

            return data

        except Exception as e:
            return GetTravelInfo().run(prompt,intent, llm, amadeus)

class ClarifyIntent:
    def __init__(self):
        pass

    def run(self, prompt,intent, llm, amadeus):
        try:
          print("Inside ClarifyIntent: ", prompt)
          if(intent=="Travel Advice" or intent=="Packing And Travel Tips" or intent=="Travel Itineraries"):
            template = """1. Analyze the  user input: {prompt} to understand if user is trying to search for restaurants or attractions or trying to do something else
            2. Return your answer in only one word in lower case from following options
              1. restaurants
              2. attractions
              3. other"""
            prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)
            result = llm_chain.run(prompt)
            if "restaurants" in result:
              print("Restaurants")
              return PointsOfInterestAgent().run(prompt,result, llm, amadeus)
            elif "attractions" in result:
              print("attractions")
              return PointsOfInterestAgent().run(prompt,result, llm, amadeus)
            
          if(intent=="Flight And Boarding Information" or intent=="Cost And Budget Planning" or intent=="Booking Assistance" or intent=="Travel Advice" or intent=="Accommodation Details"):
            template = """1. Analyze the  user input: {prompt} to understand if user is trying to search for flights or hotels or trying to do something else
            2. Return your answer in only one word in lower case from following options  1. "flights" or 2. "hotels" or 3. "other"'"""
            prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)
            result = llm_chain.run(prompt)
            if "flights" in result:
              print("Flights")
              return FlightAgent().run(prompt,intent, llm, amadeus)
            elif "hotels" in result:
              print("Hotels")
              return HotelAgent().run(prompt,intent, llm, amadeus)
            else:
              print("others.", result)
              return GetTravelInfo().run(prompt,intent, llm, amadeus)
          else:
            print("others..")
            return GetTravelInfo().run(prompt,intent, llm, amadeus)
        except Exception as e:
          print("others...")
          return GetTravelInfo().run(prompt,intent, llm, amadeus)


class GetTravelInfo:
    def __init__(self):
        pass

    def run(self, prompt, intent, llm, amadeus):
        try:
          print("Inside GetTravelInfo: ", prompt)

          template = """1. Carefully analyze the user's prompt: `{prompt}` to identify the specific information they are requesting.
          2. First try to retrieve real-world data that matches their request and format it as a JSON response.
          3. If real-world data is not available or is difficult to access, generate realistic and detailed sample data in JSON format.
          4. Prioritize returning real-world data whenever possible.
          5. Make sure the structure and content of your JSON response is relevant to the user's request. """
          prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
          llm_chain = LLMChain(llm=llm, prompt=prompt_template)
          result = llm_chain.run(prompt)
           # Store API response in a dictionary
          # result_dict = {"Data_Retrieved": result}
          # Convert dictionary to JSON string
          result_json = json.dumps(result)
          formatted_response = json_to_plain_text(result_json, 0)# ... (Process data as needed, e.g., extract names, ratings, etc.) ...

          return result_json
        except Exception as e:
          return f"An error occurred: {e}"


# --- Planning Agent ---
class TravelPlanningAgent:
    def __init__(self, llm, amadeus):
        self.llm = llm
        self.amadeus = amadeus
        self.agents = {
            "Flight And Boarding Information": ClarifyIntent(),
            "Accommodation Details": ClarifyIntent(),
            "Cancellation Fees": GetTravelInfo(),
            "Cost And Budget Planning": ClarifyIntent(),
            "Booking Assistance":ClarifyIntent(),
            "Packing And Travel Tips": ClarifyIntent(),
            "Customer Support Or Troubleshooting":GetTravelInfo(),
            "Travel Documentation": GetTravelInfo(),
            "Travel Itineraries": ClarifyIntent(),
            "Travel Advice":ClarifyIntent(),
            "Contact Details":GetTravelInfo()
        }

    def run(self, intent, prompt):
      if intent in self.agents:
        return self.agents[intent].run(prompt, intent, self.llm, self.amadeus)
      else:
          return "Invalid intent."
      
if __name__ == "__main__":
   
    amadeus = Client(
        client_id="QBSBUsh1p6DwLRtF2KpDavFECGYvZ4Ui",
        client_secret="fL4XnNYLFeitqO4I",
    )
   
    # token = os.getenv("GROQ_API_KEY")
    # llm = ChatGroq(
    #             temperature=0.1,
    #             model_name="mixtral-8x7b-32768",
    #             groq_api_key=token,
    #             max_tokens=1024
    #         )
    
    token = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo", openai_api_key=token)
   
    agent = TravelPlanningAgent(llm, amadeus)
    # response = agent.run("Travel Advice", "I am looking for a veg restaurant in Mumbai")
    response = agent.run("Travel Advice", "I am planning a trip to Mumbai")
    print(response)
      

# agent = TravelPlanningAgent()


