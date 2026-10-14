"""Gradio demo for the Chapter 4 CedarBridge LangGraph ReAct agent."""
import gradio as gr
from agent_core import CedarBridgeReAct, PERSONAS

agent = None

def get_agent():
    global agent
    if agent is None:
        agent = CedarBridgeReAct()
    return agent

def chat(message, history, persona_label, prompt_style, shot_style, placement, use_default_config, temperature):
    if not message.strip():
        return "", history, []
    persona_key = next((k for k, p in PERSONAS.items() if p.label == persona_label), "friendly")
    try:
        result = get_agent().respond(
            message, history, persona=persona_key,
            prompt_style=prompt_style, shot_style=shot_style, placement=placement,
            overrides={} if use_default_config else {"temperature": temperature})
        response = result["answer"]
        trace = result["trace"]
    except Exception as exc:
        response = f"Unable to complete request: {type(exc).__name__}: {str(exc)[:180]}"
        trace = [{"error": type(exc).__name__}]
    # Do not put raw tool arguments or sensitive lead data in trace.
    h = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
    return "", h, trace

with gr.Blocks(title="CedarBridge Learning — C4 ReAct Agent") as demo:
    gr.Markdown("# CedarBridge Learning · ReAct Agent")
    gr.Markdown("**Fictional** Lebanese online tutoring (Math, Physics, beginner Python). "
                "Persona and prompt experiments use the **same business facts**. "
                "Leads and feedback are saved **locally** only after applicable validation.")
    with gr.Row():
        persona = gr.Dropdown(choices=[p.label for p in PERSONAS.values()],
                              value=PERSONAS["friendly"].label, label="Persona")
        prompt_style = gr.Dropdown(choices=["direct", "brief_steps"], value="direct", label="Reasoning instruction")
        shot_style = gr.Dropdown(choices=["few_shot", "zero_shot"], value="few_shot", label="Examples in instructions")
        placement = gr.Dropdown(choices=["system", "in_message"], value="system", label="Persona instruction placement")
    with gr.Row():
        use_default_config = gr.Checkbox(value=True, label="Use each persona’s configured temperature (recommended for comparisons)")
        temperature = gr.Slider(0.0, 1.5, value=0.6, step=0.1, label="Custom temperature (only when above box unchecked)")
    chatbot = gr.Chatbot( height=450, label="Conversation")
    message = gr.Textbox(placeholder="Ask about tuition, tutoring availability, or leave feedback...", label="Your message")
    trace = gr.JSON(label="Manual ReAct trace (arguments and private data omitted)")
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")
    send.click(chat, [message, chatbot, persona, prompt_style, shot_style, placement, use_default_config, temperature],
               [message, chatbot, trace])
    message.submit(chat, [message, chatbot, persona, prompt_style, shot_style, placement, use_default_config, temperature],
                   [message, chatbot, trace])
    clear.click(lambda: ([], [], ""), outputs=[chatbot, trace, message])
    gr.Examples(["Do you teach Chemistry?", "How much is a 60-minute Math lesson?",
                 "I want Grade 10 Physics tutoring.",
                 "My experience with scheduling was confusing. Please record my feedback.",
                 "Please contact me about Grade 10 Math. My name is Maya Demo, "
                 "email maya.demo@example.com, and I consent to be contacted."], inputs=message)

if __name__ == "__main__":
    demo.launch()
