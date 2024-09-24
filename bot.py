import re

def simple_chatbot(user_input):
    # Convert input to lowercase for easier matching
    user_input = user_input.lower()

    # Define rules and responses
    rules = [
        (r'\b(hi|hello|hey)\b', "Hello! How can I help you today?"),
        (r'\bhow are you\b', "I'm doing well, thank you for asking! How about you?"),
        (r'\bweather\b', "I'm sorry, I don't have real-time weather information. You might want to check a weather app or website for that."),
        (r'\b(bye|goodbye)\b', "Goodbye! Have a great day!"),
        (r'\bname\b', "My name is ChatBot. It's nice to meet you!"),
        (r'\bhelp\b', "I'm a simple chatbot. I can respond to greetings, answer how I'm doing, and say goodbye. What would you like to chat about?"),
    ]

    # Check input against rules and return appropriate response
    for pattern, response in rules:
        if re.search(pattern, user_input):
            return response

    # Default response if no rule matches
    return "I'm not sure how to respond to that. Can you try rephrasing or ask me something else?"

def main():
    print("ChatBot: Hello! I'm a simple chatbot. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("ChatBot: Goodbye! Have a great day!")
            break
        response = simple_chatbot(user_input)
        print("ChatBot:", response)

if __name__ == "__main__":
    main()