import { useState, useEffect } from "react";
import schedules from "../data/schedules.json";

const STORAGE_KEY = "krishibee_plot_v1";
const DAY_MS = 24 * 60 * 60 * 1000;

function loadPlot() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function savePlot(plot) {
  try {
    if (plot) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(plot));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    return;
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

function taskStatus(task, sowing, today, done) {
  if (done) return { key: "done", label: "Done" };
  const start = addDays(sowing, task.startDay);
  const end = addDays(sowing, task.endDay);
  if (today > end) return { key: "passed", label: "Window passed" };
  if (today >= start) return { key: "due", label: "Due now" };
  const daysLeft = Math.round((start - today) / DAY_MS);
  return { key: "upcoming", label: `In ${daysLeft} day${daysLeft === 1 ? "" : "s"}` };
}

export default function CropTracker() {
  const [plot, setPlot] = useState(loadPlot);
  const crops = Object.keys(schedules);
  const [crop, setCrop] = useState(crops[0]);
  const [sowingDate, setSowingDate] = useState("");

  useEffect(() => {
    savePlot(plot);
  }, [plot]);

  const handleStart = (e) => {
    e.preventDefault();
    if (!sowingDate) return;
    setPlot({ crop, sowingDate, done: {} });
  };

  const toggleDone = (id) => {
    setPlot((prev) => ({ ...prev, done: { ...prev.done, [id]: !prev.done[id] } }));
  };

  const reset = () => {
    if (window.confirm("Remove this crop plan from this device?")) setPlot(null);
  };

  if (!plot) {
    return (
      <form className="tracker-setup" onSubmit={handleStart}>
        <h2>Track your crop</h2>
        <p>Pick your crop and the day you sowed. KrishiBee lays out the key operations from published schedules.</p>
        <label>
          Crop
          <select value={crop} onChange={(e) => setCrop(e.target.value)}>
            {crops.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <label>
          Sowing date
          <input type="date" value={sowingDate} onChange={(e) => setSowingDate(e.target.value)} />
        </label>
        <button type="submit" disabled={!sowingDate}>Start tracking</button>
      </form>
    );
  }

  const sowing = parseLocalDate(plot.sowingDate);
  const today = startOfToday();
  const dayCount = Math.round((today - sowing) / DAY_MS);
  const tasks = [...schedules[plot.crop].tasks].sort((a, b) => a.startDay - b.startDay);

  return (
    <div className="tracker">
      <div className="tracker-head">
        <div>
          <h2>{plot.crop}</h2>
          <p className="tracker-day">
            {dayCount < 0
              ? `Sowing planned for ${formatDate(sowing)} (in ${-dayCount} days)`
              : `Day ${dayCount} after sowing on ${formatDate(sowing)}`}
          </p>
        </div>
        <button className="link-btn" onClick={reset}>Remove plan</button>
      </div>

      <ul className="task-list">
        {tasks.map((task) => {
          const status = taskStatus(task, sowing, today, plot.done[task.id]);
          const start = addDays(sowing, task.startDay);
          const end = addDays(sowing, task.endDay);
          return (
            <li key={task.id} className={`task ${status.key}`}>
              <div className="task-top">
                <span className="task-type">{task.type}</span>
                <span className={`chip ${status.key}`}>{status.label}</span>
              </div>
              <h3>{task.title}</h3>
              <p className="task-when">
                {task.stage}: {formatDate(start)} to {formatDate(end)} (days {task.startDay} to {task.endDay})
              </p>
              <p className="task-note">{task.note}</p>
              <label className="task-check">
                <input
                  type="checkbox"
                  checked={!!plot.done[task.id]}
                  onChange={() => toggleDone(task.id)}
                />
                Mark as done
              </label>
              <ul className="task-sources">
                {task.sources.map((s) => (
                  <li key={s.url}>
                    <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>
                  </li>
                ))}
              </ul>
            </li>
          );
        })}
      </ul>

      <p className="tracker-note">
        Dates come from published schedules. Actual timing depends on variety, weather and soil, so confirm with your local KVK.
        This plan is saved only on this device.
      </p>
    </div>
  );
}