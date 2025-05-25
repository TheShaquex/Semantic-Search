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
  const [selectedModel, setSelectedModel] = useState<string>("huggingface");

  async function handleSubmit(text?: string) {
    if (isLoading) return;

    const messageText = text || question;
    const traceId = uuidv4();
    setIsLoading(true);
    setMessages(prev => [...prev, { content: messageText, role: "user", id: traceId }]);
    setQuestion("");

    try {
      const response = await axios.post("http://localhost:8000/search", {
        user_input: messageText,
        model: selectedModel,
      });

      const botMessage = {
        content: response.data.result,
        role: "assistant",
        id: uuidv4(),
        images: response.data.images || [],
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

  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      <Header />
      <div className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4" ref={messagesContainerRef}>
        {messages.length === 0 && <Overview />}
        {messages.map((message, index) => (
          <PreviewMessage key={index} message={message} />
        ))}
        {isLoading && <ThinkingMessage />}
        <div ref={messagesEndRef} className="shrink-0 min-w-[24px] min-h-[24px]" />
      </div>
      <div className="flex flex-col w-full md:max-w-3xl mx-auto px-4 pb-4 md:pb-6 space-y-2">
        <div className="flex justify-end">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="rounded-md border border-gray-700 bg-zinc-800 px-3 py-2 text-sm text-white shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition"
          >
            <option value="huggingface">🤖 Hugging Face</option>
            <option value="gemini">✨ Gemini</option>
          </select>
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
