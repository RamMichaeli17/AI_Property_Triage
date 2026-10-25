import os
import requests
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Property Triage Platform",
    page_icon="🏠",
    layout="wide"
)

st.title("🏘️ AI Property Triage Platform")
st.caption("Conversational Market Assistant & Automated Property Submission Portal")

# ==============================================================================
# Global Configuration & Endpoints
# ==============================================================================
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Default local n8n webhook URL (can be customized via the sidebar if needed)
DEFAULT_N8N_WEBHOOK = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/submit-property")

# Sidebar configuration for network flexibility
with st.sidebar:
    st.header("⚙️ System Settings")
    n8n_webhook_url = st.text_input(
        "n8n Webhook URL",
        value=DEFAULT_N8N_WEBHOOK,
        help="Target webhook URL configured in your n8n listing submission flow."
    )
    st.markdown("---")
    st.markdown("**Status Monitors:**")
    st.markdown("- Ollama Server: `localhost:11434`")
    st.markdown("- n8n Engine: `localhost:5678`")

# ==============================================================================
# Real Estate Assistant Grounding & Refusal Prompt
# ==============================================================================
OLLAMA_SYSTEM_PROMPT = """You are an expert, professional real estate assistant.
Your sole purpose is to provide factual, helpful, and concise information about real estate market concepts, property listings, and general property questions.

STRICT OPERATIONAL GUIDELINES:
1. DOMAIN BOUNDARIES: You must answer ONLY questions strictly related to real estate, housing, architecture, property features, and market terms.
2. REFUSAL POLICY: If the user asks about anything unrelated to real estate (e.g., coding, general chit-chat, cooking, politics, math), you must politely refuse:
   "I am specialized strictly as a real estate assistant and cannot assist with topics outside of property and real estate."
3. NO LEGAL OR TAX ADVICE: You must NEVER offer definitive legal or tax guidance. If asked about legal contracts, eviction proceedings, or specific property taxes, warn the user:
   "I cannot provide legal or tax advice. Please consult a licensed real estate attorney or tax advisor."
4. NO INVENTED PRICING: Do not fabricate specific property appraisals or guaranteed future values. Always remain factual and conservative."""

# ==============================================================================
# Main Application Interface Tabs
# ==============================================================================
tab_assistant, tab_submission = st.tabs([
    "💬 Conversational Assistant",
    "📋 Submit Listing"
])

# ------------------------------------------------------------------------------
# Tab 1: Conversational Assistant (Ollama Llama 3)
# ------------------------------------------------------------------------------
with tab_assistant:
    st.subheader("Real Estate Knowledge Assistant")
    st.write("Ask questions about the property market, valuation concepts, or listing criteria.")

    # Initialize chat message history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your real estate assistant. How can I help you today?"}
        ]

    # Render previous conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle incoming user message
    if user_prompt := st.chat_input("Ask a real estate question (e.g., What features increase home value?)..."):
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        ollama_payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ],
            "stream": False
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(OLLAMA_API_URL, json=ollama_payload, timeout=60)
                    if response.status_code == 200:
                        assistant_reply = response.json().get("message", {}).get("content", "")
                        st.markdown(assistant_reply)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                    else:
                        error_msg = f"Ollama returned HTTP error {response.status_code}: {response.text}"
                        st.error(error_msg)
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to Ollama. Ensure Ollama is running locally at `localhost:11434`.")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

# ------------------------------------------------------------------------------
# Tab 2: Listing Submission Tab (n8n Webhook Intake)
# ------------------------------------------------------------------------------
with tab_submission:
    st.subheader("Submit Property for Automated Triage")
    st.write("Submit listing details and photos to run the complete AI pipeline (Guardrails, RAG, and Vision Analysis).")

    with st.form("listing_submission_form"):
        agent_name = st.text_input("Listing Agent Name", placeholder="e.g., Jane Doe")

        description = st.text_area(
            "Property Description Text",
            placeholder="e.g., Spacious 3-bedroom apartment in Tel Aviv with balcony and elevator...",
            height=160
        )

        image_urls_raw = st.text_input(
            "Property Image URLs (comma-separated)",
            placeholder="e.g., https://example.com/kitchen.jpg, https://example.com/living.jpg"
        )

        submitted = st.form_submit_button("🚀 Submit to n8n Pipeline")

    if submitted:
        if not description.strip():
            st.warning("Please provide a property description before submitting.")
        else:
            image_urls_list = [url.strip() for url in image_urls_raw.split(",") if url.strip()]

            payload = {
                "agent_name": agent_name.strip() if agent_name else "Agent",
                "description": description.strip(),
                "text": description.strip(),
                "image_urls": image_urls_list
            }

            st.info("Sending submission to n8n orchestration pipeline...")
            with st.spinner("Processing listing through AI services (Guardrails, Extractor, Vision, RAG)..."):
                try:
                    response = requests.post(n8n_webhook_url, json=payload, timeout=120)

                    if response.status_code == 200:
                        result_data = response.json()
                        st.success("Triage pipeline completed successfully!")

                        if isinstance(result_data, dict):
                            # Case A: Input or Output Guardrail failed
                            if result_data.get("pass") is False or "rejected" in result_data or (
                                "reason" in result_data and result_data.get("pass") is not True
                            ):
                                st.error("⚠️ Submission Flagged by Guardrails Policy")
                                reason = result_data.get("reason", "The submission failed platform compliance rules.")
                                st.warning(f"**Rejection Reason:** {reason}")

                            # Case B: Standard successful brief returned from LLM Chain
                            elif "report" in result_data or "listing_brief" in result_data:
                                report_content = result_data.get("report") or result_data.get("listing_brief")
                                st.markdown("### 📄 Generated Listing Brief")
                                st.markdown(report_content)

                            # Case C: Generic structured JSON returned
                            else:
                                st.markdown("### 📊 Pipeline Output Data")
                                st.json(result_data)

                        elif isinstance(result_data, str):
                            st.markdown("### 📄 Pipeline Response")
                            st.markdown(result_data)

                    else:
                        st.error(f"n8n Webhook returned error HTTP {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"Failed to connect to n8n at `{n8n_webhook_url}`. Make sure n8n is running and listening.")
                except requests.exceptions.Timeout:
                    st.error("Pipeline timed out. Ensure all backend microservices are responding.")
                except Exception as e:
                    st.error(f"Error processing response: {str(e)}")