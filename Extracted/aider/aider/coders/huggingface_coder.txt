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

        prompt_to_hide = """Act as an expert software developer, specifically trained on the Mixtral model.
Take requests for changes to the supplied code.
If the request is ambiguous, ask questions.

Once you understand the request, you MUST:
1. List the files you need to modify. Only suggest changes to *read-write* files. Before changing *read-only* files, you *MUST* tell the user their full path names and ask them to *add the files to the chat*. End your reply and wait for their approval.
2. Think step-by-step and explain the needed changes with a numbered list of short sentences.
3. Describe each change with a *SEARCH/REPLACE block* per the examples below.

All changes to files must use this *SEARCH/REPLACE block* format.

# Example conversation:

## USER: These are the *read-write* files:

mathweb/flask/app.py
```python
from flask import Flask

app = Flask(__name__)

# ... (rest of the file content)

ASSISTANT: Ok.
USER: Change get_factorial() to use math.factorial
ASSISTANT:
To make this change, we need to modify mathweb/flask/app.py to:
Import the math package.
Remove the existing factorial() function.
Update get_factorial() to call math.factorial instead.
Here are the SEARCH/REPLACE blocks:
mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE

mathweb/flask/app.py
<<<<<<< SEARCH
def factorial(n):
    # ... (function definition)
=======
>>>>>>> REPLACE
mathweb/flask/app.py
<<<<<<< SEARCH
    return str(factorial(n))
=======
    return str(math.factorial(n))
>>>>>>> REPLACE
"""

        system_reminder = """Every *SEARCH/REPLACE block* must use this format:
The file path alone on a line, eg: main.py
The opening fence and code language, eg: ```python
The start of search block: <<<<<<< SEARCH
A contiguous chunk of lines to search for in the existing source code
The dividing line: =======
The lines to replace into the source code
The end of the replace block: >>>>>>> REPLACE
The closing fence: ```
Every SEARCH section must EXACTLY MATCH the existing source code, character for character, including all comments, docstrings, etc.
Include ALL the code being searched and replaced!
Only SEARCH/REPLACE files that are read-write.
To move code within a file, use 2 SEARCH/REPLACE blocks: 1 to delete it from its current location, 1 to insert it in the new location.
If you want to put code in a new file, use a SEARCH/REPLACE block with:
A new file path, including dir name if needed
An empty SEARCH section
The new file's contents in the REPLACE section
You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
"""

        # Check if the response starts with "ASSISTANT:"
        if generated_text and generated_text.startswith(prompt_to_hide):
            # If it does, remove the prompt from the generated_text
            generated_text = generated_text[len(prompt_to_hide):].strip()
        if generated_text and generated_text.startswith(system_reminder):
            # If it does, remove the system_reminder from the generated_text
            generated_text = generated_text[len(system_reminder):].strip()
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


# import json
# import requests
# from dotenv import load_dotenv
# import re
# from transformers import AutoTokenizer
# from .shared_utils import find_original_update_blocks  # For parsing diffs
# from .huggingface_prompts import HuggingFacePrompts
# from .editblock_coder import do_replace
# from .base_coder import Coder
# from aider.commands import Commands


# class HuggingFaceCoder(Coder):
#     def __init__(self, *args, **kwargs):
#         filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['edit_format', 'skip_model_availabily_check']}

#         self.io = kwargs.get('io')
#         self.root = kwargs.get('root')
#         self.verbose = kwargs.get('verbose')
#         self.main_model = kwargs.get('main_model')
#         self.dry_run = kwargs.get('dry_run')
#         self.cur_messages = []
#         self.done_messages = []
#         self.abs_fnames = set()
#         self.commands = Commands(self.io, self)
#         self.gpt_prompts = HuggingFacePrompts()
#         self.summarizer_thread = None
#         model_id = kwargs.get('main_model')
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.tokenizer = tokenizer
#         self.partial_response_function_call = {}
#         self.hide_assistant_response = False  # Add this line

#         api_key = kwargs.get('huggingface_api_key')

#         self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
#         self.headers = {"Authorization": f"Bearer {api_key}"}

#     def send(self, messages, model=None, functions=None):
#         if not model:
#             model = self.main_model.name

#         self.partial_response_content = ""
#         self.partial_response_function_call = dict()

#         user_messages = [msg for msg in messages if msg['role'] == 'user']
#         assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']

#         last_user_message = messages[1] if messages and messages[1]['role'] == 'user' else None 

#         prompt = ""
#         prompt += self.fmt_system_prompt(self.gpt_prompts.main_system) + "\n"
#         prompt += self.fmt_system_prompt(self.gpt_prompts.system_reminder) + "\n"
#         prompt += self.format_prompt(user_messages) 

#         response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt, "parameters": {"max_length": 16384}})
#         response_json = response.json()

#         if isinstance(response_json, list):
#             generated_text = response_json[0].get("generated_text")
#         else:
#             generated_text = response_json.get("generated_text")

#         prompt_to_hide = """Act as an expert software developer, specifically trained on the Mixtral model.
#         Take requests for changes to the supplied code.
#         If the request is ambiguous, ask questions.

#         Once you understand the request, you MUST:
#         1. List the files you need to modify. Only suggest changes to *read-write* files. Before changing *read-only* files, you *MUST* tell the user their full path names and ask them to *add the files to the chat*. End your reply and wait for their approval.
#         2. Think step-by-step and explain the needed changes with a numbered list of short sentences.
#         3. Describe each change with a *SEARCH/REPLACE block* per the examples below.

#         All changes to files must use this *SEARCH/REPLACE block* format.

#         # Example conversation:

#         ## USER: These are the *read-write* files:

#         mathweb/flask/app.py
#         \\```python
#         from flask import Flask

#         app = Flask(__name__)

#         # ... (rest of the file content)

#         ASSISTANT: Ok.
#         USER: Change get_factorial() to use math.factorial
#         ASSISTANT:
#         To make this change, we need to modify mathweb/flask/app.py to:
#         Import the math package.
#         Remove the existing factorial() function.
#         Update get_factorial() to call math.factorial instead.
#         Here are the SEARCH/REPLACE blocks:
#         mathweb/flask/app.py
#         <<<<<<< SEARCH
#         from flask import Flask
#         =======
#         import math
#         from flask import Flask
#         >>>>>>> REPLACE

#         mathweb/flask/app.py
#         <<<<<<< SEARCH
#         def factorial(n):
#             # ... (function definition)
#         =======
#         >>>>>>> REPLACE
#         mathweb/flask/app.py
#         <<<<<<< SEARCH
#             return str(factorial(n))
#         =======
#             return str(math.factorial(n))
#         >>>>>>> REPLACE
#         \\```"""  # Replace with appropriate prompt

#         system_reminder = """Every *SEARCH/REPLACE block* must use this format:
#         The file path alone on a line, eg: main.py
#         The opening fence and code language, eg: \\```python
#         The start of search block: <<<<<<< SEARCH
#         A contiguous chunk of lines to search for in the existing source code
#         The dividing line: =======
#         The lines to replace into the source code
#         The end of the replace block: >>>>>>> REPLACE
#         The closing fence: \\```
#         Every SEARCH section must EXACTLY MATCH the existing source code, character for character, including all comments, docstrings, etc.
#         Include ALL the code being searched and replaced!
#         Only SEARCH/REPLACE files that are read-write.
#         To move code within a file, use 2 SEARCH/REPLACE blocks: 1 to delete it from its current location, 1 to insert it in the new location.
#         If you want to put code in a new file, use a SEARCH/REPLACE block with:
#         A new file path, including dir name if needed
#         An empty SEARCH section
#         The new file's contents in the REPLACE section
#         You are diligent and tireless!
#         You NEVER leave comments describing code without implementing it!
#         You always COMPLETELY IMPLEMENT the needed code!"""

#         if generated_text and generated_text.startswith(prompt_to_hide):
#             generated_text = generated_text[len(prompt_to_hide):].strip()

#         if generated_text and generated_text.startswith(system_reminder):
#             generated_text = generated_text[len(system_reminder):].strip()

#         if self.repo and self.auto_commits and not self.dry_run:
#             edited_files = self.update_files()  # Get the edited files (should be a set)
#             if edited_files:
#                 self.apply_updates(edited_files)  # Call apply_updates instead of apply_edits

#     def get_edits(self):
#         edits = []

#         if "```diff" in self.partial_response_content:  # Unified diff format
#             edits = list(find_original_update_blocks(self.partial_response_content))
#         else:
#             edits = self.parse_edits_custom()  # Replace with your custom parsing function

#         return edits

#     def parse_edits_custom(self):

#         edit_pattern = r"--- (.+?)\n\+\+\+ (.+?)\n@@.*@@\n(-.*?\n|\+.*?\n)+"
#         matches = re.findall(edit_pattern, self.partial_response_content)
#         edits = [(path, original, updated) for path, original, updated in matches]

#         return edits

#     def apply_edits(self, edits):
#         for path, original, updated in edits:
#             full_path = self.abs_root_path(path)
#             content = self.io.read_text(full_path)
#             try:
#                 content = do_replace(full_path, content, original, updated, self.fence)
#             except ValueError as err:
#                 self.io.tool_error(err.args[0])  # Print the error message
#                 continue  # Move on to the next edit

#             if content:
#                 self.io.write_text(full_path, content)
#                 continue
#             self.io.tool_error(f"Failed to apply edit to {path}")

#     def format_prompt(self, messages):
#         prompt = ""
#         for message in messages:
#             role = message["role"].upper()
#             content = message.get("content")
#             if content:
#                 prompt += f"{role}: {content}\n"
#         return prompt


#     def ai_output(self, content):
#         if not self.hide_assistant_response:
#             pass  # Replace with appropriate code
