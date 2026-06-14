from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from stock_agent import build_agent, query_agent


app = FastAPI(title="Stock Agent UI")
templates = Jinja2Templates(directory="templates")

agent = build_agent()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "question": "",
            "answer": None,
            "error": None,
        },
    )


@app.post("/", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    question = question.strip()

    if not question:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "question": "",
                "answer": None,
                "error": "Please enter a question.",
            },
        )

    try:
        answer = query_agent(agent, question)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "question": question,
                "answer": answer,
                "error": None,
            },
        )

    except Exception:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "question": question,
                "answer": None,
                "error": "Something went wrong while processing your question.",
            },
        )