from fastapi import FastAPI

app = FastAPI()

@app.get()
def root()
    return {status running}

@app.post(reset)
def reset()
    return {message environment reset}

@app.post(step)
def step()
    return {message step executed}