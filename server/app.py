from fastapi import FastAPI

app = FastAPI()

# ---- API Routes ----

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/reset")
def reset():
    return {"message": "environment reset"}

@app.post("/step")
def step():
    return {"message": "step executed"}


# ---- Required entry point ----

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
