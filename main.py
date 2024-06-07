from fastapi import FastAPI, HTTPException, Query, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Access the environment variable for Groq API key
groq_api_key = os.environ.get('groq_api_key')

# Initialize memory for conversation
memory = ConversationBufferWindowMemory(k=0, memory_key="chat_history", return_messages=True)

# Define request body model
class QuestionInput(BaseModel):
    user_question: str

# Define response model
class AnswerOutput(BaseModel):
    answer: str

# Hard-coded system prompt
SYSTEM_PROMPT = """
Task: Assist users in performing tasks across the web by breaking down the process into atomic subtasks.

1. Receive User Prompt:
System: Prompt the user for the task they want to accomplish on the web.

2. Task Understanding:
System: Analyze the user's prompt to understand their intent and determine the specific action to be performed.

3. Website Selection:
System: Identify the appropriate website(s) or online platform(s) where the task can be completed effectively.

4. Navigation:
System: Navigate to the selected website(s) using the relevant URLs or search queries.

5. User Authentication (if required):
System: Handle user authentication processes, such as login or account creation, to access personalized features or services on the website(s).

6. Data Input:
System: Input necessary data or parameters required to execute the task, such as search queries, form entries, or preferences.

7. Information Retrieval:
System: Retrieve relevant information from the website(s), including search results, product listings, articles, or user-generated content.

8. Data Processing:
System: Process the retrieved information to extract key details or perform necessary computations, such as filtering search results, analyzing product specifications, or summarizing textual content.

9. User Interaction:
System: Interact with the website(s) to perform actions on behalf of the user, such as clicking links, submitting forms, adding items to cart, or initiating transactions.

10. Error Handling:
System: Monitor for errors or unexpected outcomes during navigation or interaction, and handle them appropriately, such as retrying failed actions, providing error messages, or seeking user input for resolution.

11. Confirmation:
System: Verify the successful completion of the task by checking for confirmation messages, transaction receipts, or other indicators of success.

12. Feedback:
System: Provide feedback to the user, summarizing the outcome of the task and presenting relevant details or next steps for their reference.

13. User Satisfaction Monitoring:
System: Monitor user satisfaction levels through feedback mechanisms, adjusting its performance and recommendations to enhance the overall user experience.

14. Continuous Learning:
System: Incorporate user feedback and interaction data to improve its capabilities over time, ensuring ongoing optimization of task performance and user satisfaction.
"""

# Route for rendering the HTML form
@app.get("/", response_class=HTMLResponse)
async def read_form():
    html_content = """
    <html>
        <head>
            <title>Chatbot Form</title>
        </head>
        <body>
            <h2>Chatbot Question Form</h2>
            <form action="/ask/" method="post">
                <label for="user_question">Question:</label><br>
                <input type="text" id="user_question" name="user_question" value="What is the weather today?"><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """
    return html_content

# Route for answering questions
@app.post("/ask/", response_model=AnswerOutput)
async def ask_question(user_question: str = Form(...), model: str = Query('llama3-8b-8192')):
    # Construct a chat prompt template with the hard-coded system prompt
    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        "{human_input}"
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    # Create a conversation chain
    conversation = LLMChain(
        llm=ChatGroq(groq_api_key=groq_api_key, model_name=model),
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    # Generate response from the chatbot
    response = conversation.predict(human_input=user_question)

    # Save the conversation
    message = {'human': user_question, 'AI': response}
    memory.save_context({'input': user_question}, {'output': response})

    return {"answer": response}

# Run the server with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
