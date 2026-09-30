# KRISHISANGAM AI (కృషిసంగమ్ AI)
### India's Cooperative AI Network for Climate-Resilient Farming

[![Google Gemini 3.8 Flash](https://img.shields.io/badge/Gemini_API-3.8_Flash-purple.svg)](https://ai.google.dev/)
[![Firebase Firestore](https://img.shields.io/badge/Firebase-v12.19.0-orange.svg)](https://firebase.google.com/)
[![React 19 & Vite](https://img.shields.io/badge/Frontend-React_19_|_Vite-blue.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)

---

## 1. Problem
Indian agriculture faces unprecedented compounding climate shocks: unseasonal cyclonic rainfall along coastal deltas, prolonged dry spells in continental Deccan plateaus, declining soil organic carbon (<0.6%), and excessive groundwater exploitation. Smallholder farmers operate in localized data silos:
- Agricultural research stations generate breakthrough climate-adaptation practices, but findings remain trapped in academic bulletins.
- Farmers in neighboring agro-ecological zones (such as Andhra Pradesh and Telangana) share contiguous river basins and soil classifications, yet lack an interoperable, privacy-preserving mechanism to exchange verified regenerative techniques.
- Commercial advisory apps often push proprietary synthetic agrochemicals or generic uncalibrated forecasts without regard for the farmer's specific crop stage, soil nutrient profile, or moisture risk.

---

## 2. Solution
**KRISHISANGAM AI** bridges this gap as India’s first cooperative agricultural intelligence platform:
1. **Environmental Intelligence Triad**: Combines localized soil health test parameters (pH, N, P, K, Organic Carbon), agrometeorological weather intelligence with 4 risk indices (heat, waterlogging, drought, pest), and Sentinel-2 spectral vegetation telemetry (NDVI, NDWI, crop stress, and 7/30/90-day trend lines).
2. **Transparent Multi-Factor Crop Suitability Engine**: A transparent, weighted scoring model (Soil 25%, Climate 20%, Water 20%, Satellite 15%, Crop Rotation 10%, Regenerative Value 10%) that demonstrates exact factor contributions rather than black-box claims.
3. **Multimodal AI Crop Doctor**: Powered by Google Gemini 3.8 Flash for visual disease identification, providing safe biological remedies (Jeevamrutham, Pseudomonas fluorescens, Trichoderma) and cultural management with zero toxic chemical prescriptions.
4. **Trilingual Voice & Text Assistant**: Telugu-first, Hindi, and English conversational interface grounded in farm telemetry, soil moisture, and rainfall forecasts.
5. **Agri Knowledge Exchange (AKX)**: The flagship innovation. A cross-state agricultural cooperation network that calculates multi-dimensional contextual similarity across regional profiles (climate, soil, crop, water, and seasonality) between Andhra Pradesh, Telangana, Karnataka, Maharashtra, Tamil Nadu, and Odisha. Google Gemini then interprets and adapts proven agronomic practices across state lines while safeguarding farmer privacy.

---

## 3. Architecture

```
                                  +---------------------------------------+
                                  |     KRISHISANGAM AI Client (React 19) |
                                  +-------------------+-------------------+
                                                      |
                   +----------------------------------+----------------------------------+
                   |                                                                     |
         +---------v---------+                                                 +---------v---------+
         | LocalDemoStore    | (Default Hackathon Mode)                        | FirestoreDataStore| (Live Firebase)
         | - Browser Memory  |                                                 | - Cloud Firestore |
         | - Deterministic   |                                                 | - Firebase Auth   |
         | - Instant Zero-Key|                                                 | - Firebase Storage|
         +---------+---------+                                                 +---------+---------+
                   |                                                                     |
                   +----------------------------------+----------------------------------+
                                                      |
                                                      v
                                      +---------------+---------------+
                                      | Express Backend Proxy API     |
                                      | (Port 3000 / server.ts)       |
                                      +---------------+---------------+
                                                      |
      +------------------------+----------------------+-----------------------+------------------------+
      |                        |                      |                       |                        |
+-----v--------------+   +-----v--------------+ +-----v---------------+ +-----v--------------+  +-------v-------------+
| Google Gemini SDK  |   | SatelliteProvider  | | WeatherProvider     | | SoilProvider        |  | Contextual AKX      |
| - gemini-3.8-flash |   | - DemoSentinel     | | - DemoAgroWeather   | | - DemoSoilDataset   |  | Similarity Engine   |
| - Crop Doctor      |   | - EarthEngine Live | | - LiveWeather API   | | - LiveSoil Digital  |  | - Regional Profiles |
| - Telugu Assistant |   +--------------------+ +---------------------+ +---------------------+  | - Multi-factor Calc |
| - Farm Advisory    |                                                                           +---------------------+
+--------------------+
```

---

## 4. Technology Stack
- **Frontend**: React 19, TypeScript, Vite 8, Tailwind CSS v4, Lucide React icons, Motion animations.
- **Backend / Proxy**: Node.js, Express, tsx, dotenv, cross-env, rimraf.
- **AI / LLM**: `@google/genai` TypeScript SDK utilizing `gemini-3.8-flash` for multimodal vision, conversational advice, and cross-state adaptation.
- **Data Persistence**: Firebase SDK v12 (Firestore, Authentication, Storage) + LocalDemoStore dual-mode abstraction.
- **Voice Integration**: Browser Web Speech API (`webkitSpeechRecognition` & `speechSynthesis`) with Bhashini-ready abstraction layer.

---

## 5. Google Gemini Integration
The platform uses the modern `@google/genai` SDK on the server (`server.ts` proxy):
- **Multimodal Plant Pathology (`/api/gemini/crop-doctor`)**: Analyzes plant foliage images, returning structured condition classification, scientific name, severity, confidence, immediate cultural/biological actions, and prevention guidelines.
- **Telugu/English/Hindi Farmer Assistant (`/api/gemini/assistant`)**: Grounded in specific farm parameters (crop stage, soil pH, moisture, 48-hour rain chance) to provide localized, actionable advice.
- **AI-Assisted Farm Advisory (`/api/gemini/farm-advisory`)**: Generates a 14-day agronomic schedule, soil biology guidance, and irrigation regime.
- **Cross-State Knowledge Adaptation (`/api/gemini/knowledge-adaptation`)**: Adapts practices transferred between states, detailing what to adopt, what modifications are required, and what practices should not be transferred.

---

## 6. Satellite Architecture
Defined in `src/services/providers/satelliteProvider.ts`:
- `SatelliteProvider` Interface:
  - `DemoSatelliteProvider`: Calibrated deterministic Sentinel-2 spectral model computing NDVI, EVI, NDWI, crop stress index, and historical 7/30/90-day trends.
  - `EarthEngineSatelliteProvider`: Production-ready interface for Google Earth Engine / Copernicus Sentinel-2 Hub.
- **Provenance & Labeling**: Demo data is explicitly marked as `DEMO DATA (Sentinel-2 Calibrated Model)`. Live data is marked `LIVE DATA`. Demo data is never falsely claimed as live satellite imagery.

---

## 7. Weather Architecture
Defined in `src/services/providers/weatherProvider.ts`:
- `WeatherProvider` Interface:
  - `DemoWeatherProvider`: Calibrated agrometeorological model calculating temperature, humidity, rainfall in past 24h, 7-day forecast, rain probabilities, and 4 agronomic risk indices:
    1. Heat Risk (canopy transpiration)
    2. Waterlogging Risk (soil drainage limit)
    3. Drought Risk (evaporative deficit)
    4. Pest & Disease Risk (humidity-temperature leaf wetness model)
  - `LiveWeatherProvider`: API client for open meteorological radar and agrometeorology station feeds.
- **Provenance & Labeling**: Labeled as `DEMO DATA (Agrometeorology Simulation)` in demo mode.

---

## 8. Soil Architecture
Defined in `src/services/providers/soilProvider.ts`:
- `SoilProvider` Interface:
  - `DemoSoilProvider`: Calibrated regional dataset (e.g. East Godavari Alluvial Loam: pH 6.8, N 215 kg/ha, P 22.4 kg/ha, K 295 kg/ha, OC 0.58%).
  - `LiveSoilProvider`: Digital Soil Mapping / Soil Health Portal API client.
- **Provenance & Labeling**: Labeled honestly as `Demo Soil Dataset`. Never falsely claimed as an official certified laboratory certificate.

---

## 9. Agri Knowledge Exchange (AKX)
AKX enables horizontal knowledge transfer between state nodes:
- **Regional Profiles (`stateRegionalProfiles.ts`)**: Encapsulates climate zone, annual rainfall, average and peak temperatures, dominant soil textures, pH range, water availability index, dominant crops, crop calendar, and major regional agricultural risks for AP, Telangana, Karnataka, Maharashtra, Tamil Nadu, and Odisha.
- **Transparent Mathematical Similarity Engine (`knowledgeTransferEngine.ts`)**:
  $$\text{Similarity} = 0.25 \times \text{Climate} + 0.25 \times \text{Soil} + 0.20 \times \text{Crop} + 0.20 \times \text{Water} + 0.10 \times \text{Season}$$
- **Climax Demonstration (Andhra Pradesh $\rightarrow$ Telangana)**: Shows how an AP Godavari delta pulse-rotation and water-saving practice (AWD) translates to Telangana’s semi-arid Deccan plateau, with Gemini providing the scientific adaptation rationale.

---

## 10. Privacy & Zero-PII Guarantee
Privacy is architected into the system foundation:
- **Private Data Partitioning**: Farmer names, phone numbers, village cadastral survey IDs, and GPS plot polygons reside solely in private farmer documents (`/farms/{farmId}`).
- **AKX Aggregation & Redaction**: When an insight is shared to the knowledge exchange, all personally identifiable information (PII) is permanently stripped. AKX public documents store only anonymized agronomic practices, regional soil origins, and percentage impacts.
- **Firestore Security Rules (`firestore.rules`)**: Enforces that public records cannot be created if they contain `farmerName`, `phoneNumber`, `cadastralId`, or `farmPolygon`.

---

## 11. Dual-Mode Architecture: Demo Mode vs Live Mode
| Feature | Demo Mode (Default) | Live Mode |
|---|---|---|
| **Data Store** | `LocalDemoStore` (localStorage & memory) | `FirestoreDataStore` (Cloud Firestore) |
| **Authentication** | Demo role switcher ("Continue as Demo Farmer") | Firebase Authentication |
| **Credentials Required** | Zero credentials required; fully functional | `VITE_FIREBASE_*` configuration |
| **Diagnostics Fallback** | Deterministic high-accuracy demo pathology library | Direct Gemini Multimodal live API |
| **Provenance Tag** | `DEMO DATA` | `LIVE DATA` |

---

## 12. Setup & Development

### Prerequisites
- Node.js 18+ (Node 20 or 22 recommended)
- npm or bun

### Installation
```bash
npm install
```

### Starting the Development Server
```bash
npm run dev
```
The application will launch on `http://localhost:3000`.

### Type Checking & Building
```bash
npm run lint
npm run build
```

---

## 13. Environment Variables
Defined in `.env.example`:
```env
# Required for Live Gemini features (automatically injected in AI Studio)
GEMINI_API_KEY="your-gemini-api-key"

# App URL
APP_URL="http://localhost:3000"

# Optional Live Firebase Configuration
VITE_FIREBASE_API_KEY=""
VITE_FIREBASE_AUTH_DOMAIN=""
VITE_FIREBASE_PROJECT_ID=""
VITE_FIREBASE_STORAGE_BUCKET=""
VITE_FIREBASE_MESSAGING_SENDER_ID=""
VITE_FIREBASE_APP_ID=""

# Optional Live Provider API Keys
VITE_EARTH_ENGINE_API_KEY=""
VITE_WEATHER_API_KEY=""
VITE_SOIL_API_KEY=""
```

---

## 14. How to Launch the Judge Demo
1. Open the application in your browser.
2. Click the green **"LAUNCH JUDGE DEMO"** button in the top banner or hero section.
3. The interactive 7-Step Hackathon Tour will guide you through:
   - **Step 1: Farm Foundation & Field Records** (East Godavari benchmark plot)
   - **Step 2: Soil + Weather + Satellite Intelligence** (Environmental Triad)
   - **Step 3: Transparent Weighted Crop Intelligence** (Multi-factor scoring breakdown)
   - **Step 4: Multimodal AI Crop Doctor** (Gemini 3.8 Flash plant pathology)
   - **Step 5: Telugu-First AI Farmer Assistant** (Conversational voice & text advisor)
   - **Step 6: Agri Knowledge Exchange (AKX)** (🏆 **Climax Demo**: AP $\rightarrow$ TS cooperation)
   - **Step 7: National Agricultural Network** (Interstate knowledge federation)

---

## 15. Cross-Platform & Windows Compatibility
All build and runtime scripts have been sanitized for cross-platform execution:
- Replaced Unix-specific shell calls (`rm -rf`, inline `NODE_ENV=production`) with `cross-env` and `rimraf`.
- Scripts in `package.json` run cleanly on Windows PowerShell, macOS, and Linux.

---

## 16. Known Limitations & Future Scope
- **Current Limitations**:
  - Voice recognition and synthesis currently utilize the browser Web Speech API demo fallback; full Bhashini Indic voice models require backend deployment with official Bhashini API keys.
  - Live satellite telemetry requires an active Google Earth Engine service account.
- **Future Scope**:
  - Integration with Bhashini Indic Voice API for real-time dialects of rural Rayalaseema and North Coastal Andhra.
  - On-device edge ML for offline disease diagnosis in areas without cellular connectivity.
  - Direct integration with PM-KISAN, Rythu Bharosa Kendram (RBK) digital kiosks, and national Soil Health Card databases.
