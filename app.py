import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# 1. Define the Data Structure Expectations using Pydantic
class TestCase(BaseModel):
    id: str = Field(description="Unique identifier like TC001, TC002")
    title: str = Field(description="Clear, concise title of what is being tested")
    type: str = Field(description="Happy Path, Negative, or Edge Case")
    pre_conditions: str = Field(description="Pre-requisites needed before executing the test")
    steps: List[str] = Field(description="Step-by-step actions to execute the test")
    expected_result: str = Field(description="The explicitly expected outcome")

class TestCaseSuite(BaseModel):
    test_cases: List[TestCase]

# 2. Streamlit UI Layout Setup
st.set_page_config(page_title="AI Test Case Generator", layout="wide")
st.title("🤖 AI-Powered Test Case Generator")
st.caption("Transform Requirements into Structured QA Test Suites using Groq Cloud.")

# Check the hidden cloud vault for the variable first
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = None

# Sidebar Authentication Panel Fallback
with st.sidebar:
    st.header("Authentication")
    if api_key:
        st.success("🔒 API Key loaded automatically from Streamlit Secrets vault!")
    else:
        st.warning("⚠️ GROQ_API_KEY not found in Streamlit Secrets.")
        # Provide backup text box input if secret injection hasn't been set up yet
        api_key = st.text_input("Enter your Groq API Key manually to test (gsk_...):", type="password")
        st.markdown("[Get a free Groq API key instantly](https://console.groq.com/)")

user_story = st.text_area("Paste your User Story and Acceptance Criteria here:", height=200)
generate_btn = st.button("Generate Test Suite", type="primary")

# 3. LLM Processing Pipeline
if generate_btn:
    if not api_key:
        st.error("Authentication Missing: Please paste your key in the sidebar input box to run a test.")
    elif not user_story.strip():
        st.warning("Please enter a valid user story.")
    else:
        with st.spinner("Generating test suites using ultra-fast inference endpoints..."):
            try:
                # Direct LangChain connection map pointing to the Groq processing layout
                llm = ChatOpenAI(
                    model="llama-3.3-70b-versatile",
                    openai_api_key=api_key,
                    base_url="https://groq.com",
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
