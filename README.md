# KrishiBee

A farm advisory web app for Indian farmers. It recommends a sowing depth from the soil, weather and crop at a location, answers farming questions from a small set of sourced notes, tracks a wheat crop through the season, and lets a farmer check a photo of a field problem against a published list of wheat diseases, pests and nutrient deficiencies. It shows its work throughout instead of handing back a bare number.

## Live demo

- App: https://krishi-bee.vercel.app
- API docs: https://krishibee-api.onrender.com/docs

The backend runs on Render's free tier, so the first request after a quiet period can take up to a minute while the service wakes up.

## What it does

- **Depth advisor.** Pick a crop, state, season and a point on the map. The app fetches soil and weather for that point and returns a sowing depth from a neural network, next to the rule-based depth and the agronomic safe range. Covers 8 crops in 5 states.
- **Ask KrishiBee.** A chatbot that knows which crop and field the farmer just looked up. It answers from short notes taken from a published package of practices and lists the sources under each answer. When it has no sourced answer, it says so. It understands Hindi, Hinglish and English.
- **My crop.** The farmer enters a sowing date and gets a calendar of irrigation, fertilizer, weeding, disease-scouting and harvest windows, with a done checkbox per task. Field photos are kept in a dated timeline on the phone.
- **Photo check.** Any photo in the timeline can be checked against a list of wheat problems. The app shows the closest matches with what to compare, what the package advises and where it says so.

The depth advisor covers eight crops. The chatbot, tracker and photo check currently cover wheat only, because every entry in them has to be traced to a source, and that takes real work per crop.

## The problem

Sowing depth matters more than it gets credit for. Plant a seed too shallow and it dries out or gets eaten; too deep and it never breaks the surface. Most advisory material gives a single fixed number per crop ("sow wheat 4-5cm deep") and ignores that the right depth shifts with soil texture, moisture, and weather. Small and marginal farmers in India mostly work from this kind of generic advice, or none at all. The same gap shows up later in the season, when general advice about irrigation or disease does not match the stage of the crop in front of the farmer.

## How it works

### Sowing depth

**Data pipeline.** Soil properties (texture, pH, organic carbon, bulk density, nitrogen) for 145 coordinates across Punjab, Haryana, Uttar Pradesh, Madhya Pradesh, and Rajasthan were pulled from SoilGrids. Weather climatology (temperature, humidity, rainfall, solar radiation) for the same points was pulled from NASA POWER, for both Rabi and Kharif sowing windows. Phosphorus and potassium aren't in SoilGrids, so they're estimated from texture and organic carbon, calibrated against published district-level soil fertility studies for each state rather than left as an unconstrained guess.

**Agronomic rules.** `crop_database.py` holds a base sowing-depth range for each of the 8 crops, plus continuous adjustments for texture, moisture, rain and temperature. Those rules generate every training label, and the app shows the rule-based depth beside the network. The ranges were checked against the PAU packages for Rabi 2025-26 and Kharif 2026. Five crops moved: chickpea to 10–12.5 cm, mustard to 4–5 cm, maize to 3–5 cm, pearl millet to about 2.5 cm (the ±0.5 cm band is an assumption around PAU’s single value), and rice to 2–4 cm (2–3 cm for direct seeding in dry fields, 3–4 cm in tar-wattar fields, from the basmati section). Wheat is 4–6 cm for drill sowing. Soybean was already inside 2.5–5 cm. Pigeon pea has no depth in the PAU package, so its 4–5 cm range remains general guidance. The packages are written for Punjab; the depth model still covers five states.

**Synthesized training data.** Combining the soil and weather data through the rule engine produces 1,160 labeled rows (145 locations × up to 8 season-matched crops), with small added noise so labels aren't perfectly deterministic.

**Models.** Random Forest and XGBoost baselines were trained first, both landing around MAE 0.12-0.13cm / R² 0.99 on held-out test data. A custom multi-branch neural network (separate soil, weather, and crop+location branches merged before the output layer) was trained alongside them, with a 4-way ablation study (soil-only, weather-only, crop-only, combined) to check whether soil and weather features carry predictive signal beyond crop identity alone. They do, by a modest amount that moves from seed to seed. Five seeds give a combined-network test MAE of 0.166 ± 0.018 cm (R² 0.994 ± 0.001) against 0.256 ± 0.009 cm for a crop-only network (R² 0.986 ± 0.001). The error reduction is 35.2% on average, with a standard deviation of 6.7 percentage points. The five seeds were 34.7%, 38.0%, 43.8%, 33.9% and 25.3%. Soil alone and weather alone explain little (R² 0.17 ± 0.03 and 0.40 ± 0.02). The network shipped in the app is an earlier single run, test MAE 0.151 cm, which sits inside that seed range. The tree models still edge the network on the one split in the table below. The app serves that shipped network. The five-seed weights were not deployed.

**Honesty about the data.** Every label in the training set comes from the rule engine, not from observed outcomes in a field. The model's accuracy therefore reflects how well it learned the rules, not how well those rules match reality. Real field validation (actual farms, actual germination outcomes) was out of reach for this build because of access constraints, and is listed below as a limitation.

**API.** The FastAPI backend has two prediction modes. `/predict` takes all soil and weather values by hand. `/predict_by_location` takes only a crop, state, season and GPS point, and fetches soil and weather itself. Both return the neural network's prediction next to the rule engine's prediction and the safe range, so a farmer or a developer can see when they disagree. `/predict_by_location` falls back to the nearest previously collected soil point when the live SoilGrids API is unavailable. ISRIC paused that service during development, so this was a real outage, not a hypothetical one, and the response says when cached data was used.

### Chatbot

The chatbot sends the latest prediction (crop, state, season, depth, soil texture, estimated moisture) to a hosted language model on Groq, along with up to three matching notes from a small knowledge base. Each note in `data/knowledge/` is a short paraphrase of a passage in the Punjab Agricultural University (PAU) *Package of Practices for Crops of Punjab, Rabi 2025-26*, with its page reference. Notes are retrieved with BM25 plus a small Hindi and Hinglish term map, which keeps the backend light enough for a free instance. The API returns the sources with each answer. If no note matches, the model is told to say it has no sourced answer and to point the farmer to the local Krishi Vigyan Kendra (KVK). The interface also warns up front when the current crop has no notes yet.

This design came out of failures. The first version had no knowledge base and told a wheat farmer to irrigate 7 to 10 days after sowing, while Punjab guidance puts the first irrigation at about three to four weeks. A second version, built from individual research papers, still disagreed with PAU's own timetable, so the wheat content was rebuilt from the package itself.

### Crop tracker and photo log

The wheat schedule in `frontend/src/data/schedules.json` follows PAU's tables. Irrigation windows depend on the sowing date, because the package gives different intervals for October, early and late November, and December sowings. The plan also covers urea timing, a weed-control window, yellow rust scouting from December and a harvest window, and it warns when the sowing date falls outside the timely window. Fertilizer figures are PAU's baseline for medium-fertility soil and the app tells the farmer to follow a soil test where there is one. Herbicide and fungicide product choices and doses are deliberately left out, because they depend on the weed, soil and next crop; those tasks point to the package and the KVK.

The plan is stored in the browser and photos in IndexedDB, compressed to a maximum of 1280 px. Nothing about the tracker is stored on a server, which keeps it simple and private, but it also means a plan lives on one device and clearing browser data erases it.

### Photo check

Each photo can be checked against a list of 14 wheat problems (8 diseases, 2 pests and 4 nutrient deficiencies) written from the package, in `data/problems/`. A vision model on Groq looks at the photo and may only return ids from that list. The server discards any other id, and everything shown about a match (what to compare against, what PAU advises, the source page) comes from the sourced list, not from the model. If the photo is not wheat, is unreadable, or matches nothing, the app says so. Results are ranked and carry no confidence percentage, since the model cannot produce calibrated ones.

**Pilot evaluation.** To get a measured number, 40 photos were sampled at random (fixed seed, 10 each of yellow rust, brown rust, powdery mildew and healthy leaves) from the public John Innes Centre wheat disease dataset and sent through the endpoint once, without tuning on them.

| | Result |
|---|---|
| Diseased photos, correct first answer | 20 of 30 (67%, 95% interval 49-81%) |
| Diseased photos, correct disease in top three | 21 of 30 (70%, 95% interval 52-83%) |
| Diseased photos with no match returned | 5 of 30 |
| Healthy photos with no false alarm | 9 of 10 |

By class, the first answer was right for brown rust 7 of 10, powdery mildew 6 of 10 and yellow rust 7 of 10. With ten photos per class those intervals overlap heavily, so the numbers do not show that one disease is easier than another.

This is a small pilot, not a validation. The photos come mostly from the UK and Ireland, none are from Punjab. The design's advantages are that a new problem is added by editing a sourced list, and that the system stays quiet when unsure. To repeat the evaluation, download the dataset from Zenodo, then run `scripts/sample_eval.py` and `scripts/eval_diagnose.py`.

**Trained classifiers.** ResNet18 and MobileNetV3-Small were fine-tuned on the other 959 photos from the same dataset. The 40 pilot photos were held out by file hash, and the remaining photos were split once (671 train, 144 validation, 144 test) and reused for every seed. Five seeds:

| | Held-out accuracy | Macro-F1 | Pilot, diseased first answer | Pilot, healthy |
|---|---|---|---|---|
| ResNet18 | 89.4% ± 2.2% | 0.880 ± 0.023 | 25.2 ± 1.3 of 30 | 9 of 10 on every seed |
| MobileNetV3-Small | 75.3% ± 1.9% | 0.735 ± 0.014 | 24.4 ± 1.8 of 30 | 8.8 ± 0.4 of 10 |

On this pilot the vision model’s 20 of 30 sits below both means. That comparison favours the classifiers: they trained on the same dataset the pilot photos come from, and the vision model did not. With 30 diseased photos the gap is suggestive. Abstaining below a calibrated confidence of 0.8 keeps roughly 70% of the ResNet18 test photos and puts their accuracy near 95%. Neither classifier is wired into the app. The seed reports are `data/processed/classifier_multiseed_resnet18.txt` and `data/processed/classifier_multiseed_mobilenet_v3_small.txt`.

## Tech stack

- **Data collection:** Python, SoilGrids REST API, NASA POWER API
- **ML:** scikit-learn (Random Forest), XGBoost, PyTorch (custom neural network)
- **Backend:** FastAPI, uvicorn, hosted on Render
- **Chatbot and photo check:** Groq-hosted language and vision models, BM25 retrieval over a sourced knowledge base
- **Frontend:** React (Vite), Leaflet, Axios, hosted on Vercel

## Project structure

```
KrishiBee/
├── scripts/          # data collection, synthesis, validation, training, photo-check evaluation
├── models/           # model architecture + trained weights
├── api/              # FastAPI backend: prediction, chat, retrieval, photo check
├── frontend/         # React app: depth advisor, chat, crop tracker, photo log
│   └── src/data/     # wheat schedule
├── data/
│   ├── raw/          # SoilGrids + NASA POWER output
│   ├── processed/    # synthesized + validated datasets, reports
│   ├── final/        # train/val/test splits, scaler + feature config
│   ├── knowledge/    # sourced notes for the chatbot
│   └── problems/     # sourced wheat problem list for the photo check
└── notebooks/
```

## Running locally

**Backend:**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements-deploy.txt
uvicorn api.main:app --reload --port 8000
```
The API needs only `requirements-deploy.txt`. The data and training scripts need the fuller environment (pandas, scikit-learn, XGBoost, PyTorch, matplotlib and seaborn).

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Environment variables.** Set these in the shell before starting the backend (in PowerShell: `$env:GROQ_API_KEY="..."`).

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Needed for `/chat` and `/diagnose`. Never commit it. |
| `GROQ_MODEL` | Optional chat model, default `openai/gpt-oss-120b` |
| `GROQ_VISION_MODEL` | Optional vision model, default `qwen/qwen3.8-27b` |
| `ALLOWED_ORIGINS` | Comma-separated origins allowed by CORS, default `http://localhost:5173` |
| `DIAGNOSE_DISABLE_LIMITS` | Set to `1` only for local evaluation runs |

The frontend reads `VITE_API_BASE` for the backend URL and defaults to `http://localhost:8000`.

## Results summary

| Model | Test MAE (cm) | Test R² |
|---|---:|---:|
| Random Forest | 0.131 | 0.996 |
| XGBoost | 0.133 | 0.996 |
| AgroDynamicNet (combined) | 0.151 | 0.994 |
| AgroDynamicNet (crop-only) | 0.261 | 0.985 |
| AgroDynamicNet (soil-only) | 1.500 | 0.194 |
| AgroDynamicNet (weather-only) | 1.416 | 0.376 |

The table is the single run whose weights the app loads. Five more seeds, trained without replacing those weights, are in `data/processed/nn_multiseed.txt`:

| Variant | Test MAE (cm) | Test R² |
|---|---:|---:|
| AgroDynamicNet (combined) | 0.166 ± 0.018 | 0.994 ± 0.001 |
| AgroDynamicNet (crop-only) | 0.256 ± 0.009 | 0.986 ± 0.001 |
| AgroDynamicNet (soil-only) | 1.531 ± 0.026 | 0.171 ± 0.028 |
| AgroDynamicNet (weather-only) | 1.418 ± 0.025 | 0.397 ± 0.023 |

These scores measure agreement with the rule engine, not with field outcomes. R² is higher than before the audit mostly because the crops now differ more in depth (chickpea near 11 cm, mustard near 4.5 cm). Crop identity is about 87% of Random Forest importance. Within a crop, Random Forest R² runs from 0.72 for mustard to 0.95 for maize. Before the audit, on the same split, Random Forest was 0.127 / 0.989, XGBoost 0.121 / 0.990, the combined network 0.153 / 0.985 and the crop-only network 0.208 / 0.972. Those older reports are in `data/processed/*_before_audit.txt`.

## Known limitations

- Training labels come from the rule engine, not observed field outcomes
- Phosphorus, potassium and soil moisture are estimated, not measured
- No real-world field validation of the depth model yet
- Only the depth advisor covers all 8 crops; the chatbot, tracker and photo check cover wheat only
- The wheat schedule follows PAU's recommendations for irrigated, timely-sown wheat in Punjab and does not yet ask about soil texture or variety
- The photo check was evaluated on only 40 public photos from outside India and has not been tested on Punjab fields
- Language-model answers can add details that are not in the notes. The interface shows sources and a KVK reminder with each answer, but this is a risk to keep in mind
- Chat and photo-check rate limits are kept in memory and reset when the free backend restarts, so Groq's own limits are the real backstop
- Tracker plans and photos live only in the browser on one device
- SoilGrids lookups can fall back to a cached nearby point when the external API is down

## Future work

- Grad-CAM on the ResNet18 wheat classifier
- An external test on Indian field photos, or an explicit statement that none was possible
- Sourced schedules and chatbot notes for chickpea, mustard and rice; photo check stays wheat-only until there is labelled data
- A depth interval from an ensemble and split conformal prediction, with coverage stated against the rule labels
- Putting the trained classifier inside photo check, after a memory check on the 512 MB server
- A retrieval comparison of BM25 against embeddings
- Accounts and cloud sync so a plan and its photos follow the farmer across devices
- Sourced schedules, notes and problem lists for the other seven crops
- Field validation of the depth model against real sowing outcomes
- Rain-aware reminders from a live forecast, plus local-language interface and voice input

## Data and sources

- Soil properties: SoilGrids (ISRIC World Soil Information)
- Weather climatology: NASA POWER
- Wheat schedule, knowledge notes and problem list: PAU, *Package of Practices for Crops of Punjab, Rabi 2025-26*. The package is PAU's copyright. This repository contains only short paraphrased notes with page references, not the document itself, and the notes should be checked against the current edition.
- Photo-check evaluation: John Innes Centre, *Wheat disease images (small dataset)*, Zenodo record 7573133, associated with Long et al., *Plant Pathology*, 2023 (doi:10.1111/ppa.13684). The photos are used only for evaluation and are not included here.
- Map tiles: OpenStreetMap contributors

## Author

Sourav Yadav