import unittest

class TestHuggingFaceCoder(unittest.TestCase):
    def test_send_method(self):
    # Mock the requests.post method
    with patch("requests.post") as mock_post:
        # Set up a mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"generated_text": "This is a test response."}
        mock_post.return_value = mock_response

        # Create a HuggingFaceCoder instance
        coder = HuggingFaceCoder(client=None, main_model=None, io=None)  # Pass mock objects if needed

        # Call the send method with test messages
        messages = [{"role": "user", "content": "This is a test prompt."}]
        coder.send(messages)

        # Assert that the requests.post method was called with the expected arguments
        mock_post.assert_called_once_with(
            coder.api_url, headers=coder.headers, json={"inputs": "USER: This is a test prompt.\n"}
        )

        # Assert that the generated text was extracted correctly
        self.assertEqual(coder.partial_response_content, "This is a test response.")

    def test_parse_json_response(self):
    # Mock the requests.post method
    with patch("requests.post") as mock_post:
        # Set up mock responses for different formats
        mock_response_text = MagicMock()
        mock_response_text.json.return_value = {"generated_text": "This is plain text."}

        mock_response_diff = MagicMock()
        mock_response_diff.json.return_value = {
            "generated_text": "```diff\n--- file.txt\n+++ file.txt\n@@ -1 +1 @@\n-old\n+new\n```"
        }

        mock_post.side_effect = [mock_response_text, mock_response_diff]

        # Create a HuggingFaceCoder instance
        coder = HuggingFaceCoder(client=None, main_model=None, io=None)

        # Test plain text response
        messages = [{"role": "user", "content": "Test plain text."}]
        coder.send(messages)
        self.assertEqual(coder.partial_response_content, "This is plain text.")

        # Test diff response
        messages = [{"role": "user", "content": "Test diff."}]
        coder.send(messages)
        self.assertEqual(coder.get_edits(), [("file.txt", ["-old\n", "+new\n"])])

    def test_apply_edits(self):
    # Mock the io object
    mock_io = MagicMock()

    # Create a HuggingFaceCoder instance with the mocked io
    coder = HuggingFaceCoder(client=None, main_model=None, io=mock_io)

    # Set up test edits
    edits = [("file.txt", ["-old\n", "+new\n"])]

    # Call the apply_edits method
    coder.apply_edits(edits)

    # Assert that the write_text method of the mocked io was called with the expected arguments
    mock_io.write_text.assert_called_once_with(coder.abs_root_path("file.txt"), "new\n")

    def test_auto_commit(self):
    # Mock the repo object
    mock_repo = MagicMock()

    # Create a HuggingFaceCoder instance with the mocked repo
    coder = HuggingFaceCoder(client=None, main_model=None, io=None, repo=mock_repo)

    # Set up edited files
    edited = ["file.txt"]

    # Call the auto_commit method
    coder.auto_commit(edited)

    # Assert that the commit method of the mocked repo was called
    mock_repo.commit.assert_called_once()