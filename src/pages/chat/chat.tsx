import { ChatInput } from "@/components/custom/chatinput";
import { PreviewMessage, ThinkingMessage } from "../../components/custom/message";
import { useScrollToBottom } from '@/components/custom/use-scroll-to-bottom';
import { useState } from "react";
import { message } from "../../interfaces/interfaces"
import { Overview } from "@/components/custom/overview";
import { Header } from "@/components/custom/header";
import Footer from "@/components/custom/footer";
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';

export function Chat() {
  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>("gemini"); // Changed default to gemini
  
  // Session management state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [maxMemory, setMaxMemory] = useState<number>(10);
  const [hasConversationHistory, setHasConversationHistory] = useState<boolean>(false);

  async function handleSubmit(text?: string) {
    if (isLoading) return;

    const messageText = text || question;
    const traceId = uuidv4();
    setIsLoading(true);
    setMessages(prev => [...prev, { content: messageText, role: "user", id: traceId }]);
    setQuestion("");

    try {
      // Prepare request with session data
      const requestData = {
        user_input: messageText,
        model: selectedModel,
        ...(sessionId && { session_id: sessionId }), // Include session_id if exists
        max_memory: maxMemory
      };

      const response = await axios.post("http://localhost:8000/search", requestData);

      // Update session info from response
      if (response.data.session_id) {
        setSessionId(response.data.session_id);
      }
      if (response.data.has_conversation_history !== undefined) {
        setHasConversationHistory(response.data.has_conversation_history);
      }

      const botMessage = {
        content: response.data.result,
        role: "assistant",
        id: uuidv4(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("API error:", error);
      setMessages((prev) => [
        ...prev,
        { content: "Sorry, something went wrong.", role: "assistant", id: uuidv4() },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  // Clear current conversation and start new session
  const handleClearSession = async () => {
    if (sessionId) {
      try {
        await axios.delete(`http://localhost:8000/session/${sessionId}`);
      } catch (error) {
        console.warn("Error clearing session:", error);
      }
    }
    
    // Reset local state
    setMessages([]);
    setSessionId(null);
    setHasConversationHistory(false);
  };



  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      <Header />
      <div className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4" ref={messagesContainerRef}>
        {messages.length === 0 && <Overview />}
        
        {/* Session Info Banner (optional - shows when there's conversation history) */}
        {hasConversationHistory && (
          <div className="mx-4 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg text-sm">
            <div className="flex items-center justify-between">
              <span className="text-blue-700 dark:text-blue-300">
                💬 Conversation in progress (Session: {sessionId?.substring(0, 8)}...)
              </span>
              <button
                onClick={handleClearSession}
                className="text-blue-600 hover:text-blue-800 dark:text-blue-400 hover:dark:text-blue-200 underline text-xs"
              >
                New Chat
              </button>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <PreviewMessage key={index} message={message} />
        ))}
        {isLoading && <ThinkingMessage />}
        <div ref={messagesEndRef} className="shrink-0 min-w-[24px] min-h-[24px]" />
      </div>
      
      <div className="flex flex-col w-full md:max-w-3xl mx-auto px-4 pb-4 md:pb-6 space-y-2">
        {/* Controls Row */}
        <div className="flex justify-between items-center gap-4">
          {/* Session Controls */}
          <div className="flex items-center gap-2 text-sm">
            {sessionId && (
              <>
                <button
                  onClick={handleClearSession}
                  className="px-3 py-1 text-xs bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-400 rounded transition"
                >
                  🗑️ New Chat
                </button>
                <span className="text-gray-500 text-xs">
                  Memory: {maxMemory} exchanges
                </span>
              </>
            )}
          </div>

          {/* Model Selection */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600 dark:text-gray-400">Memory:</label>
            <select
              value={maxMemory}
              onChange={(e) => setMaxMemory(Number(e.target.value))}
              className="rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-zinc-800 px-2 py-1 text-xs text-gray-900 dark:text-white"
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={15}>15</option>
              <option value={20}>20</option>
            </select>
            
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="rounded-md border border-gray-700 bg-zinc-800 px-3 py-2 text-sm text-white shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition"
            >
              <option value="gemini">✨ Gemini</option>
              <option value="huggingface">🤖 Hugging Face</option>
            </select>
          </div>
        </div>

        <ChatInput
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
      <Footer />
    </div>
  );
};