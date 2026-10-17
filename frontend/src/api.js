import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export const getCrops = () => axios.get(`${API_BASE}/crops`).then((r) => r.data.crops);

export const predictByLocation = (payload) =>
  axios.post(`${API_BASE}/predict_by_location`, payload).then((r) => r.data);

export const sendChat = (payload) =>
  axios.post(`${API_BASE}/chat`, payload).then((r) => r.data);

export const getChatCoverage = () =>
  axios.get(`${API_BASE}/chat/coverage`).then((r) => r.data.crops);

export const diagnosePhoto = (payload) =>
  axios.post(`${API_BASE}/diagnose`, payload, { timeout: 90000 }).then((r) => r.data);