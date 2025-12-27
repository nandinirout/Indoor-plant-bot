import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
import io

# 1. Page Configuration (CHANGED TITLE HERE)
st.set_page_config(page_title="Indoor Plant Bot", page_icon="🌿", layout="wide")

# 2. Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please check your .env file.")
    st.stop()

# 3. Initialize the Client
@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)

client = get_client()

# 4. Load Knowledge Base
def load_knowledge_base():
    if os.path.exists("my_plant_data.txt"):
        with open("my_plant_data.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "No knowledge base found."

knowledge_base = load_knowledge_base()

# 5. System Instruction (CHANGED NAME & PERSONA HERE)
system_instruction = f"""
You are a friendly "Indoor Plant Bot".
Your goal is to help users with indoor plant care, identifying plants from photos, and store stock.

CONTEXT FROM STORE (Your "Memory"):
-------------------
{knowledge_base}
-------------------

STRICT GUIDELINES:
1. **CHECK STORE DATA FIRST:** Always check the "CONTEXT FROM STORE" above. If the user asks about *your* prices or *your* stock, use that info.
2. **USE GOOGLE SEARCH:** If the user asks about:
   - Real-time info (e.g., "Current weather").
   - General plant facts not in your text file.
   ...then use your Google Search tool to find the answer.
3. **IMAGE DIAGNOSIS:** If the user Uploads an Image:
   - Identify the plant name.
   - Diagnose diseases.
   - Suggest care tips for indoor environments.
4. Always mention if you found the info from the "Store Files" or "Online Search".
"""

# 6. Initialize Chat Session
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash", 
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )

# 7. Sidebar for Image Upload
with st.sidebar:
    st.header("📸 Plant Doctor")
    st.write("Upload a photo to identify a plant or diagnose a disease.")
    uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "png", "jpeg"])
    
    image_part = None
    if uploaded_file:
        # Show the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Plant", use_container_width=True)
        
        # Convert to bytes for Gemini
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        image_bytes = img_byte_arr.getvalue()
        
        # Create the image part for the API
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=uploaded_file.type
        )

# 8. Main Chat UI (CHANGED TITLE HERE)
st.title("🌿 Indoor Plant Bot")
st.caption("Ask about indoor plants or Upload a Photo!")

# Display previous messages
for message in st.session_state.chat_session._curated_history:
    role = "user" if message.role == "user" else "assistant"
    with st.chat_message(role):
        if message.parts[0].text:
            st.markdown(message.parts[0].text)

# 9. Main Chat Input
user_input = st.chat_input("Ask a question about this plant...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
        if image_part:
            st.info("Attached an image for analysis.")

    try:
        if image_part:
            response = st.session_state.chat_session.send_message([user_input, image_part])
        else:
            response = st.session_state.chat_session.send_message(user_input)
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
            
    except Exception as e:

        st.error(f"An error occurred: {e}")
