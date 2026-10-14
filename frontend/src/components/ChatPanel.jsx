import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { sendChat } from "../api";

const STARTERS = [
  "When should I give the first irrigation?",
  "What are the irrigation stages after the first one?",
  "How much water should the first irrigation have?",
];

function describeChatError(err) {
  if (!err.response) {
    return "Can't reach the server. Please try again.";
  }
  if (err.response.status === 429) {
    return err.response.data?.detail || "The assistant is busy right now. Please wait a minute and try again.";
  }
  if (err.response.status === 503) {
    return "The assistant isn't available right now.";
  }
  return err.response.data?.detail || "Something went wrong. Please try again.";
}

export default function ChatPanel({ context }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async (text) => {
    const message = text.trim();
    if (!message || loading) return;
    const history = messages
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content.slice(0, 1400) }));
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setInput("");
    setError(null);
    setLoading(true);
    try {
      const data = await sendChat({ message, history, context });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply, sources: data.sources },
      ]);
    } catch (err) {
      setError(describeChatError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    send(input);
  };

  return (
    <div className="chat-panel">
      <h3>Ask KrishiBee</h3>

      <div className="chat-log">
        {messages.length === 0 && (
          <div className="chat-starters">
            {STARTERS.map((s) => (
              <button key={s} type="button" onClick={() => send(s)} disabled={loading}>
                {s}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            {m.role === "assistant" ? <ReactMarkdown>{m.content}</ReactMarkdown> : <p>{m.content}</p>}
            {m.sources?.length > 0 && (
              <ul className="chat-sources">
                {m.sources.map((s) => (
                  <li key={s.title}>
                    <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}

        {loading && <div className="chat-msg assistant typing">Thinking...</div>}
        {error && <div className="chat-error">{error}</div>}
        <div ref={endRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about irrigation, crop stages..."
          maxLength={1000}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>

      <p className="chat-note">
        Answers draw on a small set of published sources. Confirm quantities with your local KVK.
      </p>
    </div>
  );
}