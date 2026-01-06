import streamlit as st
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow

load_dotenv()

# Initialize LLM using the new Google GenAI SDK
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

st.set_page_config(page_title="Ad Copy Generator", layout="centered")
st.title("🚀 Agentic Ad Copy Generator")
st.subheader("Generates *real* ad copy with an agent + tools")

with st.form("ad_form"):
    product_name = st.text_input("Product Name")
    product_description = st.text_area("Product Description")
    problem = st.text_area("Problem it solves")
    usp = st.text_input("USP (Optional)")

    age_group = st.selectbox("Age Group", ["18-24", "25-34", "35-44", "45-54", "55+"])
    gender = st.selectbox("Target Gender", ["All Genders", "Male", "Female", "Non-binary"])
    goal = st.selectbox("Campaign Goal", ["Lead Generation", "Sales", "Brand Awareness", "Website Visits"])
    tone = st.selectbox("Tone", ["Friendly", "Professional", "Fun"])

    submitted = st.form_submit_button("Generate Ad Copies")

if submitted:

    # ========== FUNCTION CALL TRACKER ==========
    function_call_log = []  # Tracks all function calls
    
    # Build the shared context for prompts
    context = f"""
Product: {product_name}
Description: {product_description}
Problem: {problem}
USP: {usp}
Audience: {age_group}, {gender}
Goal: {goal}
Tone: {tone}
"""

    # Define the ad generator function WITH TRACKING
    def generate_ad(platform: str) -> str:
        """Generate a ready-to-post ad copy for the specified platform."""
        # LOG THE FUNCTION CALL
        call_time = datetime.now().strftime("%H:%M:%S")
        function_call_log.append({
            "function": "generate_ad",
            "platform": platform,
            "time": call_time
        })
        print(f"✅ FUNCTION CALLED: generate_ad(platform='{platform}') at {call_time}")
        
        prompt = (
            f"Create a READY-TO-POST ad copy for {platform}.\n"
            f"{context}"
            "\nConstraints:\n"
            "- Only return the ad text (no explanation)\n"
            "- Strong call-to-action\n"
        )
        result = llm.complete(prompt)
        return result.text.strip()

    # Wrap it in a FunctionTool
    ad_tool = FunctionTool.from_defaults(
        fn=generate_ad,
        name="generate_ad",
        description="Generate an ad copy given the platform (Facebook, Instagram, LinkedIn, or Google Ads)"
    )

    # Create the agent using the new AgentWorkflow API
    workflow = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[ad_tool],
        llm=llm,
        system_prompt=(
            "You are an ad generation agent. "
            "Use the generate_ad function to produce ad copy for different platforms. "
            "When asked to generate an ad for a platform, call the generate_ad function with that platform name."
        ),
    )

    platforms = ["Facebook", "Instagram", "LinkedIn", "Google Ads"]
    results = {}

    async def generate_all_ads():
        for p in platforms:
            response = await workflow.run(user_msg=f"Generate an ad for {p}")
            results[p] = str(response)
        return results

    with st.spinner("🔥 Generating ad copies…"):
        asyncio.run(generate_all_ads())

    # ========== DISPLAY FUNCTION CALL LOG ==========
    st.markdown("---")
    st.markdown("### 🔍 Function Call Log")
    if function_call_log:
        st.success(f"✅ **generate_ad** was called **{len(function_call_log)}** times!")
        for call in function_call_log:
            st.write(f"• `{call['function']}(platform='{call['platform']}')` at {call['time']}")
    else:
        st.error("❌ **generate_ad** was NOT called! The agent did not use the tool.")
    st.markdown("---")

    st.success("✅ Ads Generated!")
    for plat, txt in results.items():
        with st.expander(f"{plat} Ad Copy"):
            st.write(txt)

    combined = "\n\n".join(f"--- {plat} ---\n{txt}" for plat, txt in results.items())
    st.download_button("📥 Download Ads", combined, "ads.txt", "text/plain")
