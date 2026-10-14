import { useState, useEffect } from "react";
import schedules from "../data/schedules.json";
import PhotoLog from "./PhotoLog";
import { clearPhotos } from "../photoStore";

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

function taskStatus(task, resolved, today, done) {
  if (task.reference) return { key: "reference", label: "Reference" };
  if (done) return { key: "done", label: "Done" };
  if (resolved.lastDay && today > resolved.lastDay) return { key: "passed", label: "Window passed" };
  if (today >= resolved.start) return { key: "due", label: task.ongoing ? "Keep checking" : "Due now" };
  const daysLeft = Math.round((resolved.start - today) / DAY_MS);
  return { key: "upcoming", label: `In ${daysLeft} day${daysLeft === 1 ? "" : "s"}` };
}

function describeWindow(resolved) {
  if (!resolved.days) return `from ${formatDate(resolved.start)} onward`;
  const [a, b] = resolved.days;
  if (a === b) return `${formatDate(resolved.start)} (day ${a})`;
  return `${formatDate(resolved.start)} to ${formatDate(resolved.shownEnd)} (days ${a} to ${b})`;
}

function sowingWarning(template, sowing) {
  const w = template.sowingWindow;
  if (!w) return null;
  const key = keyFromDate(sowing);
  if (key >= keyFromMmDd(w.from) && key <= keyFromMmDd(w.to)) return null;
  return w.message;
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

  const reset = async () => {
    if (!window.confirm("Remove this crop plan and its photos from this device?")) return;
    await clearPhotos();
    setPlot(null);
  };

  if (!plot || !schedules[plot.crop]) {
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

  const template = schedules[plot.crop];
  const sowing = parseLocalDate(plot.sowingDate);
  const today = startOfToday();
  const dayCount = Math.round((today - sowing) / DAY_MS);
  const tasksById = Object.fromEntries(template.tasks.map((t) => [t.id, t]));
  const rows = template.tasks
    .map((task) => ({ task, resolved: resolveTask(task, sowing, tasksById) }))
    .sort((a, b) => a.resolved.start - b.resolved.start);
  const warning = sowingWarning(template, sowing);

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

      {warning && <p className="tracker-warning">{warning}</p>}

      <ul className="task-list">
        {rows.map(({ task, resolved }) => {
          const status = taskStatus(task, resolved, today, plot.done[task.id]);
          const trackable = !task.reference && !task.ongoing;
          return (
            <li key={task.id} className={`task ${status.key}`}>
              <div className="task-top">
                <span className="task-type">{task.type}</span>
                <span className={`chip ${status.key}`}>{status.label}</span>
              </div>
              <h3>{task.title}</h3>
              <p className="task-when">{task.stage}: {describeWindow(resolved)}</p>
              <p className="task-note">{task.note}</p>
              {trackable && (
                <label className="task-check">
                  <input
                    type="checkbox"
                    checked={!!plot.done[task.id]}
                    onChange={() => toggleDone(task.id)}
                  />
                  Mark as done
                </label>
              )}
              <ul className="task-sources">
                {task.sources.map((s) => (
                  <li key={s.title}>
                    {s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : s.title}
                  </li>
                ))}
              </ul>
            </li>
          );
        })}
      </ul>

      <PhotoLog dayCount={dayCount} />

      <p className="tracker-note">
        Dates come from published schedules. Actual timing depends on variety, weather and soil, so confirm with your local KVK.
        This plan is saved only on this device.
      </p>
    </div>
  );
}