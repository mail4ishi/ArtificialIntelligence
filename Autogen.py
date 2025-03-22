import autogen
from transformers import pipeline
from typing import List, Dict, Optional
from autogen.agentchat import ConversableAgent

# Load a text generation pipeline using the lightweight GPT-2 model from Hugging Face.
# The model is configured to run on CPU with a defined padding token ID.
model_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    device="cpu",
    pad_token_id=50256,
)

class CustomAssistantAgent(autogen.AssistantAgent):
    """Custom assistant agent that uses a local Hugging Face model instead of a remote LLM service."""

    def generate_reply(self, messages, sender, config):
        """Generate a reply using the Hugging Face GPT-2 model based on the latest message content."""
        prompt = messages[-1]["content"]
        response = model_pipeline(prompt, max_new_tokens=50, do_sample=True, truncation=True)
        return response[0]['generated_text']


class MyGroupChat(autogen.GroupChat):
    def __init__(self, agents, messages, max_round=10):
        super().__init__(agents, messages, max_round=max_round)

    def select_speaker(self, last_speaker: ConversableAgent, selector: ConversableAgent, messages: Optional[List[Dict]] = None, agents: Optional[List[ConversableAgent]] = None) -> ConversableAgent:
        """Select the next speaker by rotating through the agent list in round-robin fashion."""
        next_speaker_idx = (self.agents.index(last_speaker) + 1) % len(self.agents)
        return self.agents[next_speaker_idx]


def manual_conversation(agents, initial_message, max_rounds=5):
    """Drive the conversation manually for a set number of rounds, starting from an initial user message."""
    messages = [{"role": "user", "content": initial_message}]
    current_speaker = agents[0]

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        print(f"{current_speaker.name}:")

        reply = current_speaker.generate_reply(messages=messages, sender="User", config=current_speaker.llm_config)
        print(reply)

        messages.append({"role": "assistant", "content": reply})

        next_speaker_index = (agents.index(current_speaker) + 1) % len(agents)
        current_speaker = agents[next_speaker_index]


def main():
    config_list = [
        {
            "model": "gpt2",
            #"api_key": "YOUR_API_KEY",  #Not used here. Placeholder required for compatibility with AutoGen config structure
        }
    ]

    llm_config = {"config_list": config_list, "seed": 42}

    # Create a user proxy agent to simulate a user in the chat with optional code execution support
    user_proxy = autogen.UserProxyAgent(
        name="UserAgent",
        system_message="You are a user interacting with a project management AI assistant.",
        code_execution_config={"work_dir": "coding"},
    )

    # Create an assistant agent using the custom Hugging Face-based agent class
    assistant = CustomAssistantAgent(name="AssistantAgent", llm_config=llm_config)

    # Create a project manager agent responsible for task decomposition and delegation
    project_manager = CustomAssistantAgent(
        name="ProjectManagerAgent",
        system_message="You are a project manager. Break down user queries into tasks and delegate them.",
        llm_config=llm_config
    )

    # Create a software engineer agent responsible for implementing solutions and writing code
    software_engineer = CustomAssistantAgent(
        name="SoftwareEngineerAgent",
        system_message="You are a software engineer. Implement solutions for given tasks and provide code snippets.",
        llm_config=llm_config
    )

    agents = [user_proxy, project_manager, assistant, software_engineer]

    # Start the simulated conversation between agents with an initial request
    manual_conversation(agents, initial_message="I need an API to manage tasks for a to-do list application.", max_rounds=5)


if __name__ == "__main__":
    main()
