import streamlit as st
import json
import re
from pydantic import BaseModel, Field, field_validator
from typing import List, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# 1. Define Structured Pydantic Framework with Normalization
class TestCase(BaseModel):
    id: str = Field(description="Unique identifier like TC001, TC002")
    title: str = Field(description="Clear, concise title of what is being tested")
    type: str = Field(description="Happy Path, Negative, or Edge Case")
    pre_conditions: Union[str, List[str]] = Field(description="Pre-requisites needed before executing the test")
    steps: List[str] = Field(description="Step-by-step actions to execute the test")
    expected_result: str = Field(description="The explicitly expected outcome")

    @field_validator('pre_conditions', mode='before')
    @classmethod
    def format_pre_conditions(cls, value):
        if isinstance(value, list):
            return " ".join([str(item).strip() for item in value if item])
        return str(value).strip()

class TestCaseSuite(BaseModel):
    test_cases: List[TestCase]

# 2. Main Streamlit Layout Rendering Frame
st.set_page_config(page_title="AI Test Case Generator", layout="wide")
st.title("AI-Powered Test Case Generator")

st.caption("Resilient parsing pipeline configured to intercept and repair token-limit cutoffs.")

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
        st.markdown("[Get a free Groq API key instantly](https://groq.com)")

user_story = st.text_area(
    "Paste your User Story and Acceptance Criteria here:",
    height=200,
    placeholder="As a logged-in user..."
)
generate_btn = st.button("Generate Test Suite", type="primary")

# 3. Stream Text Parsing Execution Pipeline with Automated Truncation Repair
if generate_btn:
    if not api_key:
        st.error("Authentication Missing: Please paste your key in the sidebar input box to run a test.")
    elif not user_story.strip():
        st.warning("Please enter a valid user story.")
    else:
        with st.spinner("Generating test suites using high-speed text extraction layers..."):
            try:
                # Instantiate stable baseline model
                llm = ChatGroq(
                    model="qwen/qwen3.8-27b",
                    groq_api_key=api_key,
                    temperature=0.1,
                    max_tokens=1000
                )

                # Explicit JSON configuration layout requested via System Core Prompts
                system_instruction = (
                    "You are an elite QA Automation Engineer. Analyze the user story and generate a thorough test suite "
                    "containing happy paths, negative tests, and edge cases. You must format your response strictly as a single "
                    "valid JSON object with a key 'test_cases' containing an array of test case objects. Each test case must have keys: "
                    "'id', 'title', 'type', 'pre_conditions', 'steps' (array of strings), and 'expected_result'. "
                    "Output ONLY raw JSON code. Do not include markdown blocks, conversational preamble, or tail text."
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_instruction),
                    ("human", "{user_story}")
                ])

                chain = prompt | llm
                raw_response = chain.invoke({"user_story": user_story})
                text_content = raw_response.content.strip()

                # Clean markdown blocks if the model wrapped them
                if "```json" in text_content:
                    text_content = text_content.split("```json")[1].split("```")[0].strip()
                elif "```" in text_content:
                    text_content = text_content.split("```")[1].split("```")[0].strip()

                # Robust Fallback Parsing Repair Logic: Fix structural cuts at token walls
                if not text_content.endswith("}"):
                    st.warning("⚠️ The AI hit a strict model token length limit. Repairing the JSON structure to recover completed test cases...")
                    # Find the last completed test case block item entry
                    last_valid_object_index = text_content.rfind("}")
                    if last_valid_object_index != -1:
                        # Slice text content right up to the end of the last complete object array element
                        text_content = text_content[:last_valid_object_index + 1]
                        # Safely close off the open outer array bounds and dictionary block tags
                        text_content = text_content + "]}"
                    else:
                        # Backup search parameters if objects are completely truncated
                        last_valid_array_index = text_content.rfind("]")
                        if last_valid_array_index != -1:
                            text_content = text_content[:last_valid_array_index + 1] + "}"

                # Parse text string data cleanly into functional dictionaries
                parsed_json = json.loads(text_content)
                
                # Push data structural layouts through validation framework check strings
                suite_data = TestCaseSuite.model_validate(parsed_json)

                st.success(f"Generated {len(suite_data.test_cases)} Test Cases successfully!")
                
                # Render clean expandable information cards on screen
                for tc in suite_data.test_cases:
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
                    data=json.dumps(suite_data.model_dump(), indent=2,ensure_ascii=True)
                )
            except json.JSONDecodeError as jde:
                st.error("Could not parse the model generation text bounds. Please try running the generation loop step again.")
            except Exception as e:
                st.error(f"An unexpected tracking execution error occurred: {str(e)}")
