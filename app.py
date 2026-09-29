import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

# 1. Define Structured Pydantic Scheme Layout 
class TestCase(BaseModel):
    id: str = Field(description="Unique identifier like TC001, TC002")
    title: str = Field(description="Clear, concise title of what is being tested")
    type: str = Field(description="Happy Path, Negative, or Edge Case")
    pre_conditions: str = Field(description="Pre-requisites needed before executing the test")
    steps: List[str] = Field(description="Step-by-step actions to execute the test")
    expected_result: str = Field(description="The explicitly expected outcome")

class TestCaseSuite(BaseModel):
    test_cases: List[TestCase]

# 2. Main Streamlit Layout Rendering Frame
st.set_page_config(page_title="AI Test Case Generator", layout="wide")
st.title("🤖 AI-Powered Test Case Generator")
st.caption("Powered by ultra-fast open-weight models via official Groq libraries.")

# Pull the key securely from the hidden workspace environment 
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = None

# Sidebar Fallback Authentication Box Setup
with st.sidebar:
    st.header("Authentication")
    if api_key:
        st.success("🔒 API Key loaded from Streamlit Secrets vault!")
    else:
        st.warning("⚠️ GROQ_API_KEY not found in Streamlit Secrets.")
        api_key = st.text_input("Enter your Groq API Key manually (gsk_...):", type="password")
        st.markdown("[Get a free Groq API key instantly](https://console.groq.com/)")

user_story = st.text_area(
    "Paste your User Story and Acceptance Criteria here:",
    height=200,
    placeholder="As a logged-in user..."
)
generate_btn = st.button("Generate Test Suite", type="primary")

# 3. Native Groq Execution Pipeline
if generate_btn:
    if not api_key:
        st.error("Authentication Missing: Please paste your key in the sidebar input box to run a test.")
    elif not user_story.strip():
        st.warning("Please enter a valid user story.")
    else:
        with st.spinner("Generating test suites using ultra-fast native inference..."):
            try:
                # Instantiate native ChatGroq instance safely to eliminate 405 routing conflicts
                llm = ChatGroq(
                    model="eta-llama/llama-4-scout-17b-16e-instruct",
                    groq_api_key=api_key,
                    temperature=0.1
                )
                
                parser = PydanticOutputParser(pydantic_object=TestCaseSuite)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an elite QA Automation Engineer. Analyze the user story and generate a thorough test suite containing happy paths, negative tests, and edge cases.\n{format_instructions}"),
                    ("human", "{user_story}")
                ])

                chain = prompt | llm | parser
                response = chain.invoke({
                    "user_story": user_story,
                    "format_instructions": parser.get_format_instructions()
                })

                st.success(f"Generated {len(response.test_cases)} Test Cases successfully!")
                
                # Render Test Cards dynamically inside UI blocks
                for tc in response.test_cases:
                    with st.expander(f"**[{tc.id}]** - {tc.title} ({tc.type})"):
                        st.markdown(f"**Pre-conditions:** {tc.pre_conditions}")
                        st.markdown("**Steps:**")
                        for idx, step in enumerate(tc.steps, 1):
                            st.markdown(f"{idx}. {step}")
                        st.markdown(f"**Expected Result:** *{tc.expected_result}*")
                
                st.download_button(
                    label="Download Test Suite (JSON)",
                    file_name="test_suite.json",
                    mime="application/json",
                    data=json.dumps(response.model_dump(), indent=2)
                )
            except Exception as e:
                st.error(f"An unexpected tracking execution error occurred: {str(e)}")
