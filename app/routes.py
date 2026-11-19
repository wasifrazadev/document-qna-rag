import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


# Plain `def` (not async) so FastAPI runs these blocking calls in a threadpool
@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    rag = request.app.state.rag

    try:
        answer, sources = rag.ask(req.question)
    except Exception:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    return AskResponse(answer=answer, sources=sources)
