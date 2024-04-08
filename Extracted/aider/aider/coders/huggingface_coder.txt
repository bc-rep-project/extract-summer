import json
import requests
from dotenv import load_dotenv
import re
from transformers import AutoTokenizer
from .shared_utils import find_original_update_blocks  # For parsing diffs
from .huggingface_prompts import HuggingFacePrompts
from .editblock_coder import do_replace
from .base_coder import Coder
from aider.commands import Commands


class HuggingFaceCoder(Coder):
    def __init__(self, *args, **kwargs):
        # Filter out unnecessary arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['edit_format', 'skip_model_availabily_check']}

        self.io = kwargs.get('io')
        self.root = kwargs.get('root')
        self.verbose = kwargs.get('verbose')
        self.main_model = kwargs.get('main_model')
        self.dry_run = kwargs.get('dry_run')
        self.cur_messages = []
        self.done_messages = []
        self.abs_fnames = set()
        self.commands = Commands(self.io, self)
        self.gpt_prompts = HuggingFacePrompts()
        self.summarizer_thread = None
        model_id = kwargs.get('main_model')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tokenizer
        # super().__init__(self, *args, **kwargs)
        self.partial_response_function_call = {}
        self.hide_assistant_response = False  # Add this line

        api_key = kwargs.get('huggingface_api_key')

        # Hugging Face API URL and headers
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def send(self, messages, model=None, functions=None):
        if not model:
            model = self.main_model.name

        self.partial_response_content = ""
        self.partial_response_function_call = dict()

        # Separate user and assistant messages
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']

        # Store the last user message
        last_user_message = messages[1] if messages and messages[1]['role'] == 'user' else None 

        # Create the prompt with system prompts and user messages
        prompt = ""
        prompt += self.fmt_system_prompt(self.gpt_prompts.main_system) + "\n"
        prompt += self.fmt_system_prompt(self.gpt_prompts.system_reminder) + "\n"
        prompt += self.format_prompt(user_messages) 

        # Send the request to the Hugging Face API and get response
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt, "parameters": {"max_length": 16384}})
        response_json = response.json()

        # Extract generated text, handling potential format variations
        if isinstance(response_json, list):
            generated_text = response_json[0].get("generated_text")
        else:
            generated_text = response_json.get("generated_text")

        # Check if the response starts with "ASSISTANT:"
        if generated_text and generated_text.startswith("ASSISTANT:"):
            # If it does, extract the content after the prefix
            self.partial_response_content = generated_text[len("ASSISTANT:") + 1:].strip() 
        else:
            # Otherwise, keep the full generated text
            self.partial_response_content = generated_text

        if generated_text:
            # If the response contains code edits in unified diff format, parse them
            if "```diff" in generated_text:
                edits = list(find_original_update_blocks(generated_text))
                edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
            else:
                # If the response is plain text, handle it accordingly
                # Only print the final response if there was a user message
                if last_user_message: 
                    self.io.ai_output(generated_text) 
                edited = True  # Assuming that the output implies an edit

            if self.repo and self.auto_commits and not self.dry_run:
                edited_files = self.update_files()  # Get the edited files (should be a set)
                if edited_files:
                    self.apply_updates(edited_files)  # Call apply_updates instead of apply_edits

    def get_edits(self):
        # Extract code edits from the model's response
        edits = []

        # Check for common edit formats
        if "```diff" in self.partial_response_content:  # Unified diff format
            edits = list(find_original_update_blocks(self.partial_response_content))
        else:
            # Use a custom parsing function tailored to the specific edit format
            edits = self.parse_edits_custom()  # Replace with your custom parsing function

        return edits

    def parse_edits_custom(self):
        # Implement your custom parsing logic here
        # This function should extract the code edits from self.partial_response_content
        # and return a list of edits in the format [(path, original, updated), ...]

        # Example using a simplified regular expression without re.DOTALL:
        edit_pattern = r"--- (.+?)\n\+\+\+ (.+?)\n@@.*@@\n(-.*?\n|\+.*?\n)+"
        matches = re.findall(edit_pattern, self.partial_response_content)
        edits = [(path, original, updated) for path, original, updated in matches]

        return edits

    def apply_edits(self, edits):
        # Apply the code edits to the local files
        for path, original, updated in edits:
            full_path = self.abs_root_path(path)
            content = self.io.read_text(full_path)
            try:
                content = do_replace(full_path, content, original, updated, self.fence)
            except ValueError as err:
                # Handle ValueError specifically, potentially providing feedback or logging
                self.io.tool_error(err.args[0])  # Print the error message
                continue  # Move on to the next edit

            if content:
                self.io.write_text(full_path, content)
                continue
            self.io.tool_error(f"Failed to apply edit to {path}")

    def format_prompt(self, messages):
        # Format the messages into a single prompt string
        prompt = ""
        for message in messages:
            role = message["role"].upper()
            content = message.get("content")
            if content:
                prompt += f"{role}: {content}\n"
        return prompt


    def ai_output(self, content):
        if not self.hide_assistant_response:  # Check the flag before printing
            # Existing code to print content:
            try:
                if content.startswith("```"):  # Check if it's code
                    self.console.print(content, highlight=False)  # Print code without highlighting
                else:
                    self.console.print(
                        "[bold blue]Output generated:[/]", content
                    )  # Apply styling for other content
            except Exception as e:
                self.tool_error(f"Error formatting AI output: {e}")