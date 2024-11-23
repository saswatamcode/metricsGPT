import React, { useState } from "react";
import "./App.css";

interface ChatLogEntry {
  sender: "user" | "bot";
  message: string;
  links?: PrometheusLink[];
}

interface PrometheusLink {
  url: string;
}

const App: React.FC = () => {
  const [userMessage, setUserMessage] = useState<string>("");
  const [chatLog, setChatLog] = useState<ChatLogEntry[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const sendMessage = async () => {
    if (!userMessage.trim()) return;

    const userEntry: ChatLogEntry = { sender: "user", message: userMessage };
    const newChatLog = [...chatLog, userEntry];
    setChatLog(newChatLog);
    
    const currentMessage = userMessage;
    setUserMessage("");
    setIsStreaming(true);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: currentMessage }),
      });

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No reader available");

      let botMessage = "";
      let botLinks: PrometheusLink[] = [];

      const botEntry: ChatLogEntry = { 
        sender: "bot", 
        message: "", 
        links: [] 
      };
      setChatLog([...newChatLog, botEntry]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = new TextDecoder().decode(value);
        try {
          const parsedData = JSON.parse(chunk);
          
          if (parsedData.type === "content") {
            botMessage += parsedData.data;
            setChatLog(prevLog => [
              ...prevLog.slice(0, -1),
              { sender: "bot", message: botMessage, links: botLinks },
            ]);
          } else if (parsedData.type === "prometheus_links") {
            botLinks = parsedData.data.map((url: string) => ({ url }));
          }
        } catch (e) {
          console.error("Failed to parse chunk:", e);
        }
      }

      const cleanedMessage = botMessage.replace(/<\/?PROMQL>/g, '').trim();
      setChatLog(prevLog => [
        ...prevLog.slice(0, -1),
        { sender: "bot", message: cleanedMessage, links: botLinks },
      ]);
    } catch (error) {
      console.error("Error during chat:", error);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isStreaming && userMessage.trim()) {
      sendMessage();
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>metricsGPT</h1>
        <p className="tagline">Talk to your metrics!</p>
      </header>
      <div className="chat-container">
        <div className="chat-log">
          {chatLog.map((entry, index) => (
            <div key={index} className={`chat-entry ${entry.sender}`}>
              <div className="message-container">
                <div className="avatar">
                  {entry.sender === "bot" ? "🤖" : ""}
                </div>
                <div className="chat-message">
                  <div className="sender-name">
                    {entry.sender === "bot" ? "metricsGPT" : ""}
                  </div>
                  <div className="message-content">
                    {entry.message}
                  </div>
                  {entry.links && entry.links.length > 0 && (
                    <div className="prometheus-links">
                      {entry.links.map((link, linkIndex) => (
                        <div key={linkIndex} className="link-box">
                          <button
                            className="prometheus-button"
                            onClick={() => window.open(link.url, '_blank')}
                          >
                            View in Prometheus
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            disabled={isStreaming}
          />
          <button onClick={sendMessage} disabled={isStreaming}>
            {isStreaming ? "Streaming..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
