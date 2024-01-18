from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, Form

from src.predict_model import predict_fake_real

app = FastAPI()

# Templates configuration
templates = Jinja2Templates(directory="templates")

class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Render the HTML form
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):

    # Perform prediction using your existing model
    predicted_class, confidence = predict_fake_real(text)

    # Prepare HTML response
    html_content = f"<h2>Prediction:</h2>"
    html_content += f"<p>The text is predicted as {'FAKE' if predicted_class == 0 else 'REAL'} with confidence {confidence:.2%}</p>"

    return HTMLResponse(content=html_content)

