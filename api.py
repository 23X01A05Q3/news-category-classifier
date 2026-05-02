from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import os
import sys

# Add current dir to path to import local modules
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import clean_text
from model import load_models, predict_category

app = FastAPI(title="News Classifier API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Models ───
class LoginRequest(BaseModel):
    email: str
    password: str
    username: str = ""   # optional — validated if provided

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str
    full_name: str = ""

class PredictionRequest(BaseModel):
    headline: str
    model_type: str = "lr"  # "nb" or "lr"

# ─── User Store (email -> {password_hash, username}) ───
USERS = {
    "admin@news.com": {
        "password": hashlib.sha256("news@123".encode()).hexdigest(),
        "username": "admin"
    },
    "user@news.com": {
        "password": hashlib.sha256("pass123".encode()).hexdigest(),
        "username": "user"
    },
}

# ─── Global State for Models ───
MODELS = {
    "vectorizer": None,
    "nb": None,
    "lr": None,
    "labels": None
}

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        v, nb, lr, labels = load_models()
        MODELS["vectorizer"] = v
        MODELS["nb"] = nb
        MODELS["lr"] = lr
        MODELS["labels"] = labels
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

# ─── Endpoints ───

@app.post("/login")
async def login(req: LoginRequest):
    email = req.email.strip().lower()
    user  = USERS.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="No account found with that email.")
    # If username provided, cross-validate it
    if req.username and req.username.strip().lower() != user["username"].lower():
        raise HTTPException(status_code=401, detail="Username does not match this email.")
    hashed = hashlib.sha256(req.password.encode()).hexdigest()
    if user["password"] != hashed:
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return {"status": "success", "token": "mock-jwt-token",
            "username": user["username"], "email": email}

@app.post("/register", status_code=201)
async def register(req: RegisterRequest):
    email = req.email.strip().lower()
    if email in USERS:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    if req.username in [v["username"] for v in USERS.values()]:
        raise HTTPException(status_code=409, detail="Username already taken.")
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
    USERS[email] = {
        "password": hashlib.sha256(req.password.encode()).hexdigest(),
        "username": req.username
    }
    return {"status": "created", "username": req.username, "email": email}

@app.post("/predict")
async def predict(req: PredictionRequest):
    if MODELS["vectorizer"] is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    model = MODELS["lr"] if req.model_type == "lr" else MODELS["nb"]
    
    try:
        category, prob_dict = predict_category(
            req.headline, 
            MODELS["vectorizer"], 
            model, 
            clean_text
        )
        return {
            "category": category,
            "probabilities": prob_dict,
            "model_used": req.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": MODELS["vectorizer"] is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
