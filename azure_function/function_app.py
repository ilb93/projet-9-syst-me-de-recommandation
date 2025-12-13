import os
import json
import azure.functions as func
import requests

app = func.FunctionApp()

@app.function_name(name="reco")
@app.route(route="reco", auth_level=func.AuthLevel.ANONYMOUS)
def reco(req: func.HttpRequest) -> func.HttpResponse:
    user_id = req.params.get("user_id")
    n = req.params.get("n", "5")
    model = req.params.get("model", "content")  # content | collaborative

    if not user_id:
        try:
            body = req.get_json()
            user_id = body.get("user_id")
            n = str(body.get("n", n))
            model = body.get("model", model)
        except Exception:
            pass

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "Missing user_id (query param or JSON body)."}),
            status_code=400,
            mimetype="application/json",
        )

    base_url = os.environ.get("FASTAPI_BASE_URL", "").rstrip("/")
    if not base_url:
        return func.HttpResponse(
            json.dumps({"error": "FASTAPI_BASE_URL is not set."}),
            status_code=500,
            mimetype="application/json",
        )

    try:
        fastapi_url = f"{base_url}/reco"
        r = requests.get(
            fastapi_url,
            params={"user_id": user_id, "n": n, "model": model},
            timeout=30,
        )
        return func.HttpResponse(
            r.text,
            status_code=r.status_code,
            mimetype="application/json",
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": "FastAPI call failed", "details": str(e)}),
            status_code=502,
            mimetype="application/json",
        )
