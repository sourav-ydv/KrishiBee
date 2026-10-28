import { useEffect, useState } from "react";
import schedules from "../data/schedules.json";

const FIELD_KEY = "krishibee_field_v1";
const PLOT_KEY = "krishibee_plot_v1";
const DAY_MS = 24 * 60 * 60 * 1000;

export function saveField(latitude, longitude) {
  localStorage.setItem(FIELD_KEY, JSON.stringify({ latitude, longitude }));
}

export function loadField() {
  try {
    const raw = localStorage.getItem(FIELD_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function loadPlot() {
  try {
    const raw = localStorage.getItem(PLOT_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function parseLocalDate(value) {
  const [y, m, d] = value.split("-").map(Number);
  return new Date(y, m - 1, d);
}

function startOfToday() {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), now.getDate());
}

function addDays(date, days) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate() + days);
}

function formatDate(date) {
  return date.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

function seasonKey(month, day) {
  return (month < 6 ? month + 12 : month) * 100 + day;
}

function keyFromMmDd(value) {
  const [m, d] = value.split("-").map(Number);
  return seasonKey(m, d);
}

function keyFromDate(date) {
  return seasonKey(date.getMonth() + 1, date.getDate());
}

function windowDays(task, sowing, tasksById, depth = 0) {
  if (task.sameWindowAs && depth < 3) {
    return windowDays(tasksById[task.sameWindowAs], sowing, tasksById, depth + 1);
  }
  if (task.variants) {
    const key = keyFromDate(sowing);
    const match = task.variants.find((v) => key <= keyFromMmDd(v.sowUpTo));
    const chosen = match || task.variants[task.variants.length - 1];
    return [chosen.startDay, chosen.endDay];
  }
  return [task.startDay, task.endDay];
}

function calendarStart(sowing, value) {
  const [m, d] = value.split("-").map(Number);
  const first = new Date(sowing.getFullYear(), m - 1, d);
  return first < sowing ? new Date(sowing.getFullYear() + 1, m - 1, d) : first;
}

function resolveTask(task, sowing, tasksById) {
  if (task.fromDate) {
    return { start: calendarStart(sowing, task.fromDate), shownEnd: null, lastDay: null, days: null };
  }
  const [startDay, endDay] = windowDays(task, sowing, tasksById);
  return {
    start: addDays(sowing, startDay),
    shownEnd: addDays(sowing, endDay),
    lastDay: addDays(sowing, endDay + (task.graceDays || 0)),
    days: [startDay, endDay],
  };
}

function describeWindow(resolved) {
  if (!resolved.days) return `from ${formatDate(resolved.start)} onward`;
  const [a, b] = resolved.days;
  if (a === b) return formatDate(resolved.start);
  return `${formatDate(resolved.start)} to ${formatDate(resolved.shownEnd)}`;
}

function nextOpenTask() {
  const plot = loadPlot();
  const template = plot && schedules[plot.crop];
  if (!template) return null;
  const sowing = parseLocalDate(plot.sowingDate);
  const today = startOfToday();
  const tasksById = Object.fromEntries(template.tasks.map((task) => [task.id, task]));
  const rows = template.tasks
    .map((task) => ({ task, resolved: resolveTask(task, sowing, tasksById) }))
    .sort((a, b) => a.resolved.start - b.resolved.start);

  for (const { task, resolved } of rows) {
    if (task.reference || plot.done?.[task.id]) continue;
    if (resolved.lastDay && today > resolved.lastDay) continue;
    const daysUntil = Math.round((resolved.start - today) / DAY_MS);
    return {
      title: task.title,
      type: task.type,
      when: describeWindow(resolved),
      soon: daysUntil <= 7,
    };
  }
  return null;
}

function taskLine(task, rainLikely) {
  if (!task) return null;
  const water = /irrigat/i.test(`${task.type} ${task.title}`);
  if (water && task.soon && rainLikely) {
    return `${task.title} is ${task.when}. Rain is in the forecast, so check the field before you irrigate.`;
  }
  return `Next on your plan: ${task.title}, ${task.when}.`;
}

function skyLabel(code) {
  if (code == null) return "Unknown";
  if (code === 0) return "Clear";
  if ([1, 2, 3].includes(code)) return "Cloudy";
  if ([45, 48].includes(code)) return "Fog";
  if ([51, 53, 55, 56, 57].includes(code)) return "Drizzle";
  if ([61, 63, 65, 66, 67, 80, 81, 82].includes(code)) return "Rain";
  if ([71, 73, 75, 77, 85, 86].includes(code)) return "Snow";
  if ([95, 96, 99].includes(code)) return "Thunderstorm";
  return "Mixed";
}

function shortDate(isoDate) {
  const [y, m, d] = isoDate.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
}

async function loadForecast(latitude, longitude) {
  const params = new URLSearchParams({
    latitude: String(latitude),
    longitude: String(longitude),
    daily: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
    forecast_days: "7",
    timezone: "auto",
  });
  const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
  if (!response.ok) throw new Error("forecast failed");
  const payload = await response.json();
  const daily = payload.daily || {};
  const dates = daily.time || [];
  if (dates.length < 7) throw new Error("forecast incomplete");

  const days = dates.slice(0, 7).map((isoDate, i) => ({
    date: isoDate,
    label: shortDate(isoDate),
    sky: skyLabel(daily.weather_code[i]),
    temp_min_c: daily.temperature_2m_min[i] == null ? null : Math.round(daily.temperature_2m_min[i] * 10) / 10,
    temp_max_c: daily.temperature_2m_max[i] == null ? null : Math.round(daily.temperature_2m_max[i] * 10) / 10,
    rain_mm: daily.precipitation_sum[i] == null ? null : Math.round(daily.precipitation_sum[i] * 10) / 10,
    rain_probability_pct: daily.precipitation_probability_max[i] == null ? null : Math.round(daily.precipitation_probability_max[i]),
  }));
  const rainy = days.filter((day) => (day.rain_mm || 0) >= 1 || (day.rain_probability_pct || 0) >= 50);
  return {
    days,
    rain_likely: rainy.length > 0,
    rain_line: rainy.length
      ? `Rain is likely on ${rainy.map((day) => day.label).join(", ")}.`
      : "Little or no rain is forecast for the next 7 days.",
    source: "Open-Meteo",
    note: "This is a weather forecast, not a PAU recommendation. Check the field before you irrigate.",
  };
}

export default function ForecastCard({ latitude, longitude }) {
  const saved = latitude == null ? loadField() : null;
  const lat = latitude ?? saved?.latitude;
  const lon = longitude ?? saved?.longitude;
  const [forecast, setForecast] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (lat == null || lon == null) return undefined;
    let cancelled = false;
    setLoading(true);
    setError(null);
    loadForecast(lat, lon)
      .then((data) => {
        if (!cancelled) setForecast(data);
      })
      .catch(() => {
        if (!cancelled) {
          setForecast(null);
          setError("The 7-day forecast is unavailable right now. Try again shortly.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [lat, lon]);

  if (lat == null || lon == null) {
    return (
      <section className="forecast-card">
        <h2>This week's weather</h2>
        <p>Tap your field on the Depth advisor map. The next 7 days of rain and temperature will show here.</p>
      </section>
    );
  }

  const line = forecast ? taskLine(nextOpenTask(), forecast.rain_likely) : null;

  return (
    <section className="forecast-card">
      <h2>This week's weather</h2>
      {loading && <p>Fetching the next 7 days…</p>}
      {error && <p className="error">{error}</p>}
      {forecast && (
        <>
          <p className="forecast-rain">{forecast.rain_line}</p>
          {line && <p className="forecast-task">{line}</p>}
          <ul className="forecast-days">
            {forecast.days.map((day) => (
              <li key={day.date} className={(day.rain_mm || 0) >= 1 ? "wet" : ""}>
                <span>{day.label}</span>
                <strong>{day.sky}</strong>
                <span>{day.temp_min_c}–{day.temp_max_c}°C</span>
                <span>{day.rain_mm} mm{day.rain_probability_pct != null ? ` · ${day.rain_probability_pct}%` : ""}</span>
              </li>
            ))}
          </ul>
          <p className="forecast-note">{forecast.note} Source: {forecast.source}.</p>
        </>
      )}
    </section>
  );
}