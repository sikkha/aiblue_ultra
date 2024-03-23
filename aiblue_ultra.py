import random
import gradio as gr
from openai import OpenAI
import requests
from flask import Flask, request, jsonify, make_response
import concurrent.futures
import requests
import os
import openai
from metaphor_python import Metaphor
from datetime import datetime, timedelta
import textwrap
import json 
import configparser


g_sess = None

# Read API keys from api_key.conf in the current directory or the directory specified in the environment variable
config = configparser.ConfigParser()

# The path to the config file is constructed from the API_KEY_FILE_PATH environment variable and the filename
config_file_directory = os.getenv("API_KEY_FILE_PATH", ".")
config_file_name = "api_key.conf"
config_file_path = os.path.join(config_file_directory, config_file_name)

print("config file =", config_file_path)

config.read(config_file_path)

# Fetching the API keys
METAPHOR_API_KEY = config.get('API_KEYS', 'METAPHOR_API_KEY', fallback=None)
GOOGLE_API_KEY = config.get('API_KEYS', 'GOOGLE_API_KEY', fallback=None)
MISTRAL_API_KEY = config.get('API_KEYS', 'MISTRAL_API_KEY', fallback=None)
OPENAI_API_KEY = config.get('API_KEYS', 'OPENAI_API_KEY', fallback=None)
ANTHROPIC_API_KEY = config.get('API_KEYS', 'ANTHROPIC_API_KEY', fallback=None)

def get_api_key(model):
    """Retrieve API key based on model selection."""
    keys = {
        'palm2': GOOGLE_API_KEY,
        'gemini-pro': GOOGLE_API_KEY,
        'mistral': MISTRAL_API_KEY,
        'openai': OPENAI_API_KEY,
        'anthropic': ANTHROPIC_API_KEY
    }
    return keys.get(model, None)
    

def extract_text_from_response(model, response):
    """Extract text content from the LLM response based on the model."""
    if model == 'palm2':
        # Assuming Google's response is in the format {'candidates': [{'output': '...'}]}
        return response.get('candidates', [{}])[0].get('output', '')

    elif model == 'gemini-pro':
        # Extracting text from gemini-pro response
        candidates = response.get('candidates', [])
        if candidates:
            content_parts = candidates[0].get('content', {}).get('parts', [])
            if content_parts:
                return content_parts[0].get('text', '')
        return ''

    elif model == 'mistral':
        # Assuming Mistral's response is in the format {'choices': [{'message': {'content': '...'}}]}
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')

    elif model == 'gpt4':  # Add this block for GPT-4
        # Check if the response has choices and the first choice has a message
        if hasattr(response, 'choices') and response.choices:
            first_choice = response.choices[0]
            if hasattr(first_choice, 'message'):
                # Access the content directly from the ChatCompletionMessage object
                return first_choice.message.content
        return ''
    
    elif model == 'anthropic':
        # Extracting text from Anthropic's response
        response_json = response.json()  # Convert the response to JSON
        content = response_json.get('content', [])
        if content:
            for item in content:
                if item.get('type') == 'text':
                    return item.get('text', '')
        return ''

    elif model == 'openai':
        # Assuming OpenAI's response is in the format {'choices': [{'message': {'content': '...'}}]}
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')

    return ''


def call_LLM(model, prompt):
    headers = {'Content-Type': 'application/json'}
    api_key = get_api_key(model)

    if model in ['palm2', 'gemini-pro']:
        if model == 'palm2': # For 'palm2' 
            url = f"https://generativelanguage.googleapis.com/v1beta3/models/text-bison-001:generateText?key={api_key}"
            data = {"prompt": {"text": prompt}}
        else:  # For 'gemini-pro' 
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
            data = {"contents": [{"parts": [{"text": prompt}]}]}
    elif model == 'mistral':
        url = "https://api.mistral.ai/v1/chat/completions"
        data = {"model": "mistral-medium", "messages": [{"role": "user", "content": prompt}]}
        headers['Authorization'] = f'Bearer {api_key}'

    elif model == 'gpt4':  # Changed to 'gpt4' to match your preference
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return extract_text_from_response(model, completion)

    elif model == 'anthropic': 
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = {
            #"model": "claude-3-sonnet-20240229",
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [
               {"role": "user", "content": prompt}
            ]
        }

        completion = requests.post("https://api.anthropic.com/v1/messages", headers=headers, data=json.dumps(data))
        return(extract_text_from_response(model, completion))


    elif model == 'openai':
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helper assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        headers['Authorization'] = f'Bearer {api_key}'

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_json = response.json()
        return extract_text_from_response(model, response_json)
    else:
        return f'Error: {response.status_code}, {response.text}'


def multihead_model(prompt, gradient):
    # Ensure gradient is within the 0-100 range
    gradient = max(0, min(gradient, 100))

    # Calculate weights for each model
    weight_google = gradient / 100.0
    weight_mistral = 1 - weight_google
    
    verify_result = verify_internet_rag(f"{prompt}")

    # Call the submodels
    response_gemini_pro = call_LLM('gemini-pro', verify_result)
    response_mistral = call_LLM('mistral', verify_result)

    # Prepare a refined prompt for the router model
    combined_prompt = (
        "Here are the summarized inputs from two analysis models:\n\n"
        "1. Primary Analysis (Weight: {:.0%}): {}\n\n"
        "2. Secondary Analysis (Weight: {:.0%}): {}\n\n"
        "Based on these analyses, provide a concise and definitive summary of the situation, focusing on key insights and conclusions."
        .format(weight_google, response_gemini_pro, weight_mistral, response_mistral)
    )

    # Use OpenAI as the router to process the refined prompt
    final_output = call_LLM('openai', combined_prompt)

    return final_output


def devil_advocate(prompt, gradient):
    # Ensure gradient is within the 0-100 range
    gradient = max(0, min(gradient, 100))

    # Calculate weights for Google's perspective and Mistral's counterargument
    weight_google = gradient / 100.0
    weight_mistral = 1 - weight_google
    
    verify_result = verify_internet_rag(f"{prompt}")

    # Gemini-Pro refines the initial prompt
    gemini_pro_perspective = call_LLM('gemini-pro', verify_result)

    # Prompt for the devil's advocate model (Mistral) to provide a counterargument
    devil_advocate_prompt = (
        f"Given the following perspective, provide a counterargument or alternative viewpoint:\n\n"
        f"{gemini_pro_perspective}"
    )
    devil_advocate_response = call_LLM('mistral', devil_advocate_prompt)

    # Prepare a prompt for OpenAI to synthesize a final, concise summary
    synthesis_prompt = (
        f"Initial perspective (Weight: {weight_google:.0%}): {gemini_pro_perspective}\n\n"
        f"Devil's Advocate perspective (Weight: {weight_mistral:.0%}): {devil_advocate_response}\n\n"
        "Considering these perspectives with their respective weights, provide a concise and clear summary."
    )

    # Use OpenAI to synthesize the final summary
    final_summary = call_LLM('openai', synthesis_prompt)

    return final_summary


def devil_advocate2(prompt, gradient):
    # Ensure gradient is within the 0-100 range
    gradient = max(0, min(gradient, 100))

    # Calculate weights for gemini-pro's perspective and Mistral's counterargument
    weight_gemini_pro = gradient / 100.0
    weight_mistral = 1 - weight_gemini_pro
    
    verify_result = verify_internet_rag(f"{prompt}")

    #verify_result = background_search(f"{prompt}")

    # Gemini-Pro refines the initial prompt
    gemini_pro_perspective = call_LLM('gemini-pro', f"Refine the following passage:\n\n{verify_result}")

    # Prompt for the devil's advocate model (Mistral) to provide a counterargument
    devil_advocate_prompt = (
        f"Given the following refined perspective, provide a counterargument or alternative viewpoint:\n\n"
        f"{gemini_pro_perspective}"
    )
    devil_advocate_response = call_LLM('mistral', devil_advocate_prompt)

    # Prepare a prompt for OpenAI to combine these perspectives
    combined_prompt = (
        f"Refined perspective (Weight: {weight_gemini_pro:.0%}): {gemini_pro_perspective}\n\n"
        f"Devil's Advocate perspective (Weight: {weight_mistral:.0%}): {devil_advocate_response}\n\n"
        "Considering these perspectives with their respective weights, combine them into a coherent narrative. Make it strictly within 120 words."
    )
    combined_response = call_LLM('openai', combined_prompt)
    
    # Use PaLM2 to summarize the final output
    summary_prompt = f"Base on the following information please analyze, don't do bullet.:\n\n{combined_response}"
    final_summary = call_LLM('palm2', summary_prompt)
    
    return final_summary


def verify_internet_rag(prompt):
    # Initialize OpenAI and Metaphor
    # openai.api_key = openai_api_key
    metaphor = Metaphor(METAPHOR_API_KEY)

    # Generate a search query using OpenAI
    system_message = "You are a helpful assistant that generates search queries based on user questions. Only generate one search query."
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    search_query = completion.choices[0].message.content

    # Perform a search using Metaphor
    one_week_ago = datetime.now() - timedelta(days=7)
    one_month_ago = datetime.now() - timedelta(days=30)
    date_cutoff = one_month_ago.strftime("%Y-%m-%d")
    search_response = metaphor.search(
        search_query, use_autoprompt=True, start_published_date=date_cutoff
    )

    # Extract URLs from the search response (optional)
    urls = [result.url for result in search_response.results]

    # Get content from the first search result
    contents_result = search_response.get_contents()
    content_item = contents_result.contents[0] if contents_result.contents else None

    # Generate a summary of the first search result's content
    if content_item:
        system_message_summary = "You are a helpful assistant that briefly summarizes the content of a webpage. Summarize the users input."
        completion_summary = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message_summary},
                {"role": "user", "content": content_item.extract},
            ],
        )
        summary = completion_summary.choices[0].message.content
        formatted_summary = textwrap.fill(summary, 200)
        return f"Summary for {content_item.url}:\n{content_item.title}\n{formatted_summary}"
    else:
        return "No content available for summarization."

    
def background_search(prompt, include_domains=None, start_published_date="2023-06-25"):
    # Initialize the Metaphor client
    metaphor = Metaphor(METAPHOR_API_KEY)

    # Perform the search
    search_response = metaphor.search(
        prompt,
        include_domains=include_domains,
        start_published_date=start_published_date,
    )

    # Get the contents of the search response
    contents_response = search_response.get_contents()

    # Compile and return the search results as a string
    results_str = ""
    for content in contents_response.contents:
        result = f"Title: {content.title}\nURL: {content.url}\nContent:\n{content.extract}\n"
        results_str += result + "\n"

    return results_str


def devil_advocate2_concurrent(prompt, gradient):
    # Concurrent Pre-Processing: Validate and refine the prompt independently
    with concurrent.futures.ThreadPoolExecutor() as pre_executor:
        future_verification = pre_executor.submit(verify_internet_rag, prompt)
        future_gradient_adjustment = pre_executor.submit(lambda grad: max(0, min(grad, 100)), gradient)

        # Wait for all pre-processing tasks to complete
        verify_result = future_verification.result()
        adjusted_gradient = future_gradient_adjustment.result()

    # Calculate weights based on the adjusted gradient
    weight_gemini_pro = adjusted_gradient / 100.0
    weight_mistral = 1 - weight_gemini_pro

    # Sequential Calls (Dependent): Gemini-Pro refines, then Mistral provides a counterargument
    gemini_pro_perspective = call_LLM('gemini-pro', f"Refine the following passage:\n\n{verify_result}")
    devil_advocate_prompt = f"Given the following refined perspective, provide a counterargument:\n\n{gemini_pro_perspective}"
    devil_advocate_response = call_LLM('mistral', devil_advocate_prompt)

    # Concurrent Post-Processing: Combine perspectives and summarize
    with concurrent.futures.ThreadPoolExecutor() as post_executor:
        future_combined = post_executor.submit(
            call_LLM, 'openai',
            f"Refined perspective (Weight: {weight_gemini_pro:.0%}): {gemini_pro_perspective}\n\n"
            f"Devil's Advocate perspective (Weight: {weight_mistral:.0%}): {devil_advocate_response}\n\n"
            "Combine these perspectives into a coherent narrative, within 120 words."
        )

        future_summary = post_executor.submit(
            call_LLM, 'palm2',
            f"Based on the following information, please analyze (no bullets):\n\n"
        )

        # Wait for the post-processing tasks to complete
        combined_response = future_combined.result()
        # Update the summary prompt with the actual combined response
        final_summary = post_executor.submit(call_LLM, 'palm2', f"{future_summary.result()}{combined_response}").result()

    return final_summary


def append_interaction(session_id, user_message, chatbot_response):
    # Combine the user message and chatbot response into one entry
    global g_sess
    dialogue_entry = "User: " + user_message + " | Chatbot: " + chatbot_response

    # Append the new dialogue entry to the session in the database
    # debug, this still a problem we global variable to hard fix it <<<
    session_id = str(session_id)
    if session_id is not None and "[[" in session_id:
       session_id = g_sess
    else:
       g_sess = session_id
    append_interaction_to_session(session_id, dialogue_entry)
    
    # Debugging line to monitor session_id
    print(f"Appending interaction for session_id: {session_id}")


def get_or_start_session(session_id):
    global g_sess
    if not session_id:
        # If session_id is None or empty, start a new session
        start_session_url = "http://localhost:5001/start_session"
        response = requests.get(start_session_url)
        # Ensure the response is successful and has JSON content
        if response.ok:
            session_data = response.json()
            session_id = session_data.get("session_id", "")  # Get the session_id from the response
            if g_sess is None:
               g_sess = session_id
        else:
            # Handle the case where the response is not successful
            print("Failed to start a new session. Server responded with:", response.status_code)
            # Potentially set session_id to None or handle it accordingly
            session_id = None
    return session_id


def read_recent_interactions(session_id, number_of_interactions=7):
    if not session_id:
        print("No session ID provided.")
        return None  # Or handle this case as appropriate for your application
    
    get_latest_sessions_url = "http://localhost:5001/get_latest_sessions"
    params = {'number': number_of_interactions}
    cookies = {'SessionID': session_id}
    
    try:
        response = requests.get(get_latest_sessions_url, cookies=cookies, params=params)
        if response.status_code == 200:
            session_data = response.json()
            return session_data.get('interactions', [])
        else:
            print("Failed to retrieve session data. Server responded with:", response.status_code)
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def append_interaction_to_session(session_id, new_interaction):
    append_session_url = "http://localhost:5001/update_session"  # Assuming this endpoint now appends a new interaction
    headers = {'Content-Type': 'application/json'}
    cookies = {'SessionID': session_id}
    data = {'new_interaction': new_interaction}  # Changed from 'new_history' to 'new_interaction'
    
    # Make the request to the server to append the new interaction
    print(append_session_url)
    print("Session ID before request:", session_id)
    response = requests.post(append_session_url, json=data, cookies=cookies, headers=headers)
    print(append_session_url)

    
    if response.ok:
        # Return the server's response which might include confirmation or additional info
        print("response ok")
        return response.json()
    else:
        print("response failed")
        # Handle cases where the server response isn't successful
        print("Failed to append interaction. Server responded with:", response.status_code)
        return None  # or handle error as appropriate


def engage_response(message, session_id):

    global g_sess
    # Retrieve or start a new session based on the existing user_id (from cookies)
    # debug
    session_id = get_or_start_session(session_id)
    print(session_id)

    # Retrieve the recent interactions for the session
    session_id = str(session_id)
    if session_id is not None and "[[" in session_id:
       session_id = g_sess
    else:
       g_sess = session_id
    my_history = ""
    my_history = read_recent_interactions(session_id)

    # start chatbot logic
    gradient = slider_value

    # original code change to multiple-ai
    #almighty = devil_advocate(message, gradient) 
    global ai_choice 

    # << configuration prompt

    # Read configuration prompt from aiblue.conf
    config = configparser.ConfigParser()
    config.read('aiblue.conf')

    persona = config.get('Configuration', 'persona')
    name = config.get('Configuration', 'name')
    creator = config.get('Configuration', 'creator')
    style = config.get('Configuration', 'style')
    top_priority = config.get('Configuration', 'top_priority')
    max_length = config.getint('History', 'max_length')
    store = config.get('History', 'store')
    recall = config.get('History', 'recall')
    instructions = config.get('Instructions', 'instructions')

    dialogue_data = f"""
    Proceed with the following settings as your context and configuration. Focus on answering the user's latest inquiry within the context provided:
    <dialogue>
        <configuration>
            <persona>{persona}</persona>
            <name>{name}</name>
            <creator>{creator}</creator>
            <style>{style}</style>
            <top_priority>{top_priority}</top_priority>
            <history_control>
                <max_length>{max_length}</max_length>
                <store>{store}</store>
                <recall>{recall}</recall>
            </history_control>
        </configuration>

        <encoded_dialogue_history>
            my_history
        </encoded_dialogue_history>

        <task_list>
            <task>Decode the dialogue history.</task>
            <task>Remember the user's name.</task>
            <task>Remember the chatbot's name, AI Blue.</task>
            <task>Detect and adapt to the user's language and emotional cues.</task>
            <task>Review the dialogue history for contextual relevance.</task>
            <task>For Lord33rd, activate enhanced response protocols upon hearing "Abracadabra."</task>
        </task_list>

        <latest_user_inquiry>
            {message}
        </latest_user_inquiry>
        <instructions>
            {instructions}
        </instructions>
    </dialogue>
    """
    
    # << end configuration prompt

    if ai_choice == "Solo":
    	#almighty = call_LLM("mistral", dialogue_data)  
    	#almighty = call_LLM("gemini-pro", dialogue_data)  
    	#almighty = call_LLM("gpt4", dialogue_data)  
    	#almighty = call_LLM("openai", dialogue_data)  
        almighty = call_LLM("anthropic", dialogue_data)  

    elif ai_choice == "MultiHead":
    	almighty = multihead_model(message, gradient)  

    elif ai_choice == "DevilAdvocate":
    	almighty = devil_advocate(message, gradient)  

    elif ai_choice == "Parallel":
    	almighty = devil_advocate2_concurrent(message, gradient)  

    zip_command = []
    zip_command.append("compress the following conversation into zipped encoding style, which chatbot can understand further, no need for human to understand: " + " User: " + message + " Chatbot: " + almighty)

    #zipped_output = call_LLM("gemini-pro", zip_command)  
    # Append the model's response to the history

    # Append the new interaction (user message and chatbot response) to the session
    # (Assuming 'almighty' is the chatbot's response)
    append_interaction(session_id, message, almighty)

    return almighty
    #return almighty, session_id 

   # pass

# Define a function to update the label based on the slider's value.
def update_label(value):
    slider_value = value
    return f"Slider is at {value}"


def process_selection(choice):
    # 'choice' will contain the value of the selected radio button.
    global ai_choice

    ai_choice = choice
    return f"You selected: {choice}"

app = Flask(__name__)

# Flask route to start a new session or retrieve an existing one
@app.route('/start_session', methods=['GET'])
def start_session():
    return session_manager.start_session()

# Flask route to get the current session ID or create a new one
@app.route('/get_session_id', methods=['GET'])
def get_session_id():
    return session_manager.get_session_id()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    slider_value = 70
    ai_choice = "Solo"
    with gr.Row():
        slider = gr.Slider(0, 100, step=1, value=slider_value, label="AI Blue Gradient")
        #label = gr.Label()  # Uncomment if you want to use this later.
        radio = gr.Radio(["Solo", "MultiHead", "DevilAdvocate", "Parallel"], label="Choose Your Engine", value="Solo")

    # Bind the slider to update the label.
    slider.change(update_label, inputs=[slider])

    # Output where the result will be displayed. (for debug)
    #label = gr.Label()

    #radio.change(process_selection, inputs=radio, outputs=label)
    radio.change(process_selection, inputs=radio)

    chat_demo = gr.ChatInterface(engage_response)


if __name__ == "__main__":
    demo.launch()
    print(demo.server_port)


