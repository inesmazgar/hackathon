document.addEventListener("DOMContentLoaded", function () {
    const chatContainer = document.querySelector(".chat-container");
    const userInput = document.querySelector("#user-input");
    const sendButton = document.querySelector("#send-btn");

    // Add a message to the chat
    function addMessage(message, isUser) {
        const messageElement = document.createElement("div");
        messageElement.className = isUser ? "user-message" : "bot-message";
        messageElement.textContent = message;
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to the bottom
    }

    // Send message to the Flask backend
    async function sendMessageToServer(message) {
        try {
            const response = await fetch("http://127.0.0.1:8000/chat", {

                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: message }),
            });

            if (!response.ok) {
                throw new Error("Erreur lors de la communication avec le serveur.");
            }

            const data = await response.json();
            return data.response; // The bot's response
        } catch (error) {
            console.error("Erreur :", error);
            return "Une erreur est survenue. Veuillez réessayer.";
        }
    }

    // Handle send button click
    sendButton.addEventListener("click", async function () {
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addMessage(userMessage, true); // Add user message to chat
            userInput.value = ""; // Clear input field

            const botResponse = await sendMessageToServer(userMessage); // Get response from server
            addMessage(botResponse, false); // Add bot response to chat
        }
    });

    // Allow pressing "Enter" to send a message
    userInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            sendButton.click();
        }
    });
});
async function sendFeedback(feedback) {
    const lastMessage = document.querySelector(".bot-message:last-child").textContent;

    const feedbackData = {
        message: lastMessage,
        feedback: feedback
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/feedback", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(feedbackData),
        });

        if (response.ok) {
            alert("Merci pour votre retour !");
        } else {
            alert("Impossible d'envoyer les commentaires.");
        }
    } catch (error) {
        console.error("Erreur lors de l'envoi des commentaires : ", error);
    }
}

