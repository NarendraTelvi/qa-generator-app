import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI # Groq uses the standard OpenAI client protocol compatibility

# Define structured Pydantic expectations for reliable JSON generation
class TestCase(BaseModel):
    id: str = Field(description="Unique identifier like TC001, TC002")
    title: str = Field(description="Clear, concise title of what is being tested")
    type: str = Field(description="Happy Path, Negative, or Edge Case")
    pre_conditions: str = Field(description="Pre-requisites needed before executing the test")
    steps: List[str] = Field(description="Step-by-step actions to execute the test")
    expected_result: str = Field(description="The explicitly expected outcome")

class TestCaseSuite(BaseModel):
    test_cases: List[TestCase]

st.set_page_config(page_title="Free AI Test Case Generator", layout="wide")
st.title("🤖 Free Open-Source AI Test Case Generator")
st.caption("Hosted for free on Streamlit Cloud, powered by open-source models.")

# Fetch the API key safely from Streamlit's environment settings
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.sidebar.warning("⚠️ GROQ_API_KEY not found in Streamlit Secrets. Please enter it below to test:")
    GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key manually", type="password")

user_story = st.text_area("Paste your User Story and Acceptance Criteria here:", height=200)

if st.button("Generate Test Suite", type="primary"):
    if not GROQ_API_KEY:
        st.error("Please provide a Groq API Key to proceed.")
    elif not user_story.strip():
        st.warning("Please enter a valid user story.")
    else:
        with st.spinner("Processing with cloud-hosted Llama 3..."):
            try:
                # Initialize the open-source model using Groq's high-speed free tier endpoint
                llm = ChatOpenAI(
                    model="llama3-8b-8192", 
                    openai_api_key=GROQ_API_KEY,
                    openai_api_base="https://groq.com",
                    temperature=0.1
                )
                parser = PydanticOutputParser(pydantic_object=TestCaseSuite)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an elite QA Automation Engineer. Analyze the user story and generate a thorough test suite. Output JSON strictly matching the requested schema.\n{format_instructions}"),
                    ("human", "{user_story}")
                ])

                chain = prompt | llm | parser
                response = chain.invoke({
                    "user_story": user_story, 
                    "format_instructions": parser.get_format_instructions()
                })

                st.success(f"Generated {len(response.test_cases)} Test Cases!")
                for tc in response.test_cases:
                    with st.expander(f"**[{tc.id}]** - {tc.title} ({tc.type})"):
                        st.markdown(f"**Pre-conditions:** {tc.pre_conditions}")
                        for idx, step in enumerate(tc.steps, 1):
                            st.markdown(f"{idx}. {step}")
                        st.markdown(f"**Expected Result:** *{tc.expected_result}*")
            except Exception as e:
                st.error(f"Error: {str(e)}")
