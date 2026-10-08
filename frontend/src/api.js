import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export const getCrops = () => axios.get(`${API_BASE}/crops`).then((r) => r.data.crops);

export const predictByLocation = (payload) =>
  axios.post(`${API_BASE}/predict_by_location`, payload).then((r) => r.data);