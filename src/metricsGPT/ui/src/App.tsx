import React, { useState, useRef, useEffect } from "react";
import { PulseLoader } from "react-spinners";
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
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const chatLogRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (chatLogRef.current) {
      chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    }
  }, [chatLog]);

  useEffect(() => {
    if (!isStreaming && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isStreaming]);

  const sendMessage = async () => {
    if (!userMessage.trim()) return;

    const userEntry: ChatLogEntry = { sender: "user", message: userMessage };
    const newChatLog = [...chatLog, userEntry];
    setChatLog(newChatLog);
    
    const currentMessage = userMessage;
    setUserMessage("");
    setIsStreaming(true);
    setIsThinking(true);

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
          setIsThinking(false);
          
          if (parsedData.type === "content") {
            botMessage += parsedData.data;
            const updatedEntry: ChatLogEntry = {
              sender: "bot",
              message: botMessage,
              links: botLinks
            };
            setChatLog(prevLog => [...prevLog.slice(0, -1), updatedEntry]);
          } else if (parsedData.type === "prometheus_links") {
            botLinks = parsedData.data.map((url: string) => ({ url }));
          }
        } catch (e) {
          console.error("Failed to parse chunk:", e);
        }
      }

      const finalEntry: ChatLogEntry = {
        sender: "bot",
        message: botMessage,
        links: botLinks
      };
      setChatLog(prevLog => [...prevLog.slice(0, -1), finalEntry]);
    } catch (error) {
      console.error("Error during chat:", error);
    } finally {
      setIsStreaming(false);
      setIsThinking(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isStreaming && userMessage.trim()) {
      sendMessage();
    }
  };

  const renderMessage = (message: string) => {
    const parts = message.split(/<PROMQL>|<\/PROMQL>/);
    return parts.map((part, index) => {
      // Even indices are regular text, odd indices are PromQL queries
      if (index % 2 === 0) {
        return <span key={index}>{part}</span>;
      } else {
        return (
          <pre key={index} className="promql-block">
            <code>{part}</code>
          </pre>
        );
      }
    });
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>metricsGPT</h1>
        <p className="tagline">Talk to your metrics!</p>
      </header>
      <div className="chat-container">
        <div className="chat-log" ref={chatLogRef}>
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
                    {renderMessage(entry.message)}
                    {isThinking && index === chatLog.length - 1 && entry.sender === "bot" && (
                      <div className="thinking">
                        thinking... <PulseLoader size={8} color="#666" speedMultiplier={0.7} />
                      </div>
                    )}
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
            ref={inputRef}
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            disabled={isStreaming}
          />
          <button onClick={sendMessage} disabled={isStreaming}>
            {isStreaming ? (
              <>
                Streaming... <PulseLoader size={8} color="#666" speedMultiplier={0.7} />
              </>
            ) : (
              "Send"
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
