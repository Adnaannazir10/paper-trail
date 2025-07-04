from fastapi import APIRouter
from api import routes

router = APIRouter()

router.include_router(routes.router)
