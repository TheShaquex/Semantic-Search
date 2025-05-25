export interface message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    images?: string[];
  }
