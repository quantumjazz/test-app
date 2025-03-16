document.addEventListener("DOMContentLoaded", function() {
    const toggleBtn = document.getElementById('toggle-settings');
    const settingsPanel = document.getElementById('settings-panel');
    const sendBtn = document.getElementById('send-message');
    const messageInput = document.getElementById('message-input');
    const chatArea = document.getElementById('chat-area');
  
    // Toggle the settings panel visibility
    toggleBtn.addEventListener('click', function() {
      settingsPanel.classList.toggle('collapsed');
    });
  
    // Send message on button click or when pressing Enter
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') sendMessage();
    });
  
    function sendMessage() {
      const message = messageInput.value.trim();
      if (message === "") return;
      
      appendMessage("You", message);
      messageInput.value = "";
      
      fetch('/send_message', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: message})
      })
      .then(response => response.json())
      .then(data => {
        appendMessage("Assistant", data.response);
        // Optionally update chat history based on data.chat_history
      })
      .catch(err => console.error("Error:", err));
    }
  
    function appendMessage(sender, message) {
      const messageElem = document.createElement('div');
      messageElem.classList.add('message');
      messageElem.innerHTML = `<strong>${sender}:</strong> ${message}`;
      chatArea.appendChild(messageElem);
      chatArea.scrollTop = chatArea.scrollHeight;
    }
  });
  