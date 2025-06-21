[Llama 3 Prompt Format](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)

* **<|begin_of_text|>**: This is equivalent to the BOS token
* **<|eot_id|>**: This signifies the end of the message in a turn.
* **<|start_header_id|>{role}<|end_header_id|>**: These tokens enclose the role for a particular message. The possible roles can be: system, user, assistant.
* **<|end_of_text|>**: This is equivalent to the EOS token. On generating this token, Llama 3 will cease to generate more tokens.
A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followed by the assistant header.

# Ensure ollama has been downloaded 
https://ollama.com/
run
ollama run llama3.2 in terminalll

