import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Item {
  id: number;
  title: string;
  genres: string;
  score: number;
  explanation?: string;
}

export interface RecommendationResponse {
  user_id: number;
  items: Item[];
}

export const api = {
  getUsers: async () => {
    const res = await axios.get<{ user_id: number }[]>(`${API_URL}/users?limit=50`);
    return res.data;
  },

  getRecommendations: async (userId: number) => {
    const res = await axios.get<RecommendationResponse>(`${API_URL}/recommend`, {
      params: { user_id: userId, k: 10 }
    });
    return res.data;
  },

  getHybridRecommendations: async (userId: number, query: string, alpha: number) => {
    const res = await axios.get<RecommendationResponse>(`${API_URL}/recommend_query`, {
      params: { user_id: userId, q: query, k: 10, alpha }
    });
    return res.data;
  }
};