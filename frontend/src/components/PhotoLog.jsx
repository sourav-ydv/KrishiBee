import { useState, useEffect, useRef, useCallback } from "react";
import { listPhotos, savePhoto, deletePhoto } from "../photoStore";
import { diagnosePhoto } from "../api";
import DiagnosisResult from "./DiagnosisResult";

async function compressImage(file, maxSide = 1280, quality = 0.72) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Compression failed"))),
      "image/jpeg",
      quality
    );
  });
}

function formatDay(iso) {
  return new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function describeCheckError(err) {
  if (!err.response) {
    return "Could not reach the server. It may be waking up, so please try again in a minute.";
  }
  const detail = err.response.data?.detail;
  return typeof detail === "string" ? detail : "The photo check failed. Please try again.";
}

export default function PhotoLog({ dayCount, crop }) {
  const [photos, setPhotos] = useState([]);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const cameraRef = useRef(null);
  const [checkingId, setCheckingId] = useState(null);
  const [diagnosis, setDiagnosis] = useState(null);
  const [checkError, setCheckError] = useState(null);
  const urlsRef = useRef([]);

  const refresh = useCallback(async () => {
    const items = await listPhotos();
    urlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    const withUrls = items.map((p) => ({ ...p, url: URL.createObjectURL(p.blob) }));
    urlsRef.current = withUrls.map((p) => p.url);
    setPhotos(withUrls);
  }, []);

  useEffect(() => {
    refresh();
    return () => urlsRef.current.forEach((u) => URL.revokeObjectURL(u));
  }, [refresh]);

  const handleFiles = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      for (const file of files) {
        const blob = await compressImage(file);
        await savePhoto({
          blob,
          takenAt: new Date().toISOString(),
          day: dayCount,
          note: note.trim(),
        });
      }
      setNote("");
      await refresh();
    } catch {
      setError("Could not save the photo. Try a smaller image or free up space on your device.");
    } finally {
      setBusy(false);
    }
  };

  const handleCheck = async (photo) => {
    setCheckingId(photo.id);
    setCheckError(null);
    setDiagnosis(null);
    try {
      const small = await compressImage(photo.blob, 1024, 0.7);
      const image_base64 = await blobToBase64(small);
      const data = await diagnosePhoto({
        crop,
        image_base64,
        note: photo.note || null,
        day_after_sowing: photo.day >= 0 ? photo.day : null,
      });
      setDiagnosis({ photoId: photo.id, url: photo.url, data });
    } catch (err) {
      setCheckError(describeCheckError(err));
    } finally {
      setCheckingId(null);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this photo?")) return;
    await deletePhoto(id);
    await refresh();
  };

  return (
    <section className="photo-log">
      <h3>Field photos</h3>
      <p className="photo-tip">
        Take a wide shot of the field and a close-up of a few leaves, in daylight, every week or two.
        Photos stay on this device only.
      </p>

      <div className="photo-add">
        <input
          type="text"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Optional note, e.g. yellow tips on lower leaves"
          maxLength={120}
        />
        <button type="button" onClick={() => cameraRef.current?.click()} disabled={busy}>
          {busy ? "Saving..." : "Take photo"}
        </button>
        <button type="button" className="secondary" onClick={() => inputRef.current?.click()} disabled={busy}>
          From gallery
        </button>
        <input ref={cameraRef} type="file" accept="image/*" capture="environment" hidden onChange={handleFiles} />
        <input ref={inputRef} type="file" accept="image/*" multiple hidden onChange={handleFiles} />
      </div>

      {error && <p className="photo-error">{error}</p>}

      {photos.length === 0 ? (
        <p className="photo-empty">No photos yet.</p>
      ) : (
        <ul className="photo-grid">
          {photos.map((p) => (
            <li key={p.id}>
              <a href={p.url} target="_blank" rel="noreferrer">
                <img src={p.url} alt={`Field photo from day ${p.day}`} />
              </a>
              <div className="photo-meta">
                <strong>Day {p.day}</strong> · {formatDay(p.takenAt)}
                {p.note && <span>{p.note}</span>}
              </div>
              <div className="photo-actions">
                <button
                  type="button"
                  className="check-btn"
                  onClick={() => handleCheck(p)}
                  disabled={checkingId !== null}
                >
                  {checkingId === p.id ? "Checking..." : "Check this photo"}
                </button>
                <button type="button" className="link-btn" onClick={() => handleDelete(p.id)}>
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {checkError && <p className="photo-error">{checkError}</p>}
      {diagnosis && <DiagnosisResult result={diagnosis} onClose={() => setDiagnosis(null)} />}
    </section>
  );
}