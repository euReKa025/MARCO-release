from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn
from admet_ai import ADMETModel
from rdkit import rdBase

from proper_utils import qed, penalized_logp

rdBase.DisableLog('rdApp.error')

# Initialize the model once when the application starts
model = ADMETModel()

app = FastAPI()

# Define the request body model for POST requests
class SmilesInput(BaseModel):
    smiles: str

@app.post("/predict/")
async def predict_post(smiles_input: SmilesInput):
    """
    Accepts a SMILES string via POST request body and returns ADMET predictions.
    """
    preds = model.predict(smiles=smiles_input.smiles)
    mutagenicity = preds['AMES']
    bbbp = preds['BBB_Martins']
    hia = preds['HIA_Hou']
    qed_score = qed(smiles_input.smiles)
    plogp = penalized_logp(smiles_input.smiles)
    return {"mutagenicity": mutagenicity, "bbbp": bbbp, "hia": hia, "qed": qed_score, "plogp": plogp}

@app.get("/predict/")
async def predict_get(smiles: str):
    """
    Accepts a SMILES string via GET query parameter and returns ADMET predictions.
    """
    preds = model.predict(smiles=smiles)
    mutagenicity = preds['AMES']
    bbbp = preds['BBB_Martins']
    hia = preds['HIA_Hou']
    qed_score = qed(smiles)
    plogp = penalized_logp(smiles)
    return {"mutagenicity": mutagenicity, "bbbp": bbbp, "hia": hia, "qed": qed_score, "plogp": plogp}

if __name__ == "__main__":
    host = os.environ.get("PREDICTOR_BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("ADMET_PORT", "10086"))
    uvicorn.run(app, host=host, port=port)
