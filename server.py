from io import BytesIO



from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

import imgtokens

app = FastAPI()
app.state.initialized = False
app.state.preload_model = True  # Slower startup, faster response
app.mount("/static", StaticFiles(directory="static"), name="static")


def init_ipwt():
    if app.state.initialized:
        print("Using cached model...")
        return app.state.ipwt
    print("Creating ImagePatchWordTokenizer...")
    ipwt = imgtokens.ImagePatchWordTokenizer()  # this is slow on first run
    print("Initializing model...")
    ipwt._init_model()
    print("Done initializing")
    app.state.ipwt = ipwt
    app.state.initialized = True
    return ipwt

@app.on_event("startup")
async def startup_event():
    # imgtokens.preload_models()  # this will OOM
    if app.state.preload_model:
        init_ipwt()

@app.get("/readiness")
async def readiness():
    if not app.state.preload_model:
        # More responsive
        return JSONResponse(content={"status": "ready"})
    if app.state.initialized:
        return JSONResponse(content={"status": "ready"})
    else:
        return JSONResponse(content={"status": "initializing"}, status_code=503)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...), similarity: str = "omp", num_words: int = 10):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    if num_words > 20:
        raise HTTPException(status_code=400, detail="Too many words")

    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    ipwt: imgtokens.ImagePatchWordTokenizer = init_ipwt()
    words = ipwt.process_img(img, similarity=similarity, num_words=num_words)
    fig = ipwt.draw_with_plotly(words, size=1000)
    plotly_config = {
        "displayModeBar": False,
    }
    html = fig.to_html(include_plotlyjs="cdn", config=plotly_config)
    return HTMLResponse(content=html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

