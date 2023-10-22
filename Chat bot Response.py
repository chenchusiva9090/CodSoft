def chatbot_response(user_input):
    if "hello" in user_input.lower():
        return "Hello! How can I help you?"
    elif "how are you" in user_input.lower():
        return "I'm just a computer program, but I'm here to assist you!"
    elif "bye" in user_input.lower():
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I don't understand that."

# Example usage:
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
