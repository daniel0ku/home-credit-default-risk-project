from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Credit Default Risk Prediction API")

# Define all features based on your training data
categorical_features = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'WEEKDAY_APPR_PROCESS_START',
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 
    'EMERGENCYSTATE_MODE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'
]

numeric_features = [
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
    'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 
    'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
    'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
    'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_1',
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
    'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
    'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
    'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
    'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
    'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
    'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
    'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
    'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
    'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
    'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
    'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
    'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
    'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
    'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
]

class ApplicationData(BaseModel):
    # Categorical features
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    NAME_TYPE_SUITE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    FONDKAPREMONT_MODE: Optional[str] = None
    HOUSETYPE_MODE: Optional[str] = None
    WALLSMATERIAL_MODE: Optional[str] = None
    EMERGENCYSTATE_MODE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None

    # Numeric features
    CNT_CHILDREN: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    OWN_CAR_AGE: Optional[float] = None
    FLAG_MOBIL: Optional[float] = None
    FLAG_EMP_PHONE: Optional[float] = None
    FLAG_WORK_PHONE: Optional[float] = None
    FLAG_CONT_MOBILE: Optional[float] = None
    FLAG_PHONE: Optional[float] = None
    FLAG_EMAIL: Optional[float] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    REGION_RATING_CLIENT: Optional[float] = None
    REGION_RATING_CLIENT_W_CITY: Optional[float] = None
    HOUR_APPR_PROCESS_START: Optional[float] = None
    REG_REGION_NOT_LIVE_REGION: Optional[float] = None
    REG_REGION_NOT_WORK_REGION: Optional[float] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[float] = None
    REG_CITY_NOT_LIVE_CITY: Optional[float] = None
    REG_CITY_NOT_WORK_CITY: Optional[float] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    APARTMENTS_AVG: Optional[float] = None
    BASEMENTAREA_AVG: Optional[float] = None
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = None
    YEARS_BUILD_AVG: Optional[float] = None
    COMMONAREA_AVG: Optional[float] = None
    ELEVATORS_AVG: Optional[float] = None
    ENTRANCES_AVG: Optional[float] = None
    FLOORSMAX_AVG: Optional[float] = None
    FLOORSMIN_AVG: Optional[float] = None
    LANDAREA_AVG: Optional[float] = None
    LIVINGAPARTMENTS_AVG: Optional[float] = None
    LIVINGAREA_AVG: Optional[float] = None
    NONLIVINGAPARTMENTS_AVG: Optional[float] = None
    NONLIVINGAREA_AVG: Optional[float] = None
    APARTMENTS_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = None
    YEARS_BUILD_MODE: Optional[float] = None
    COMMONAREA_MODE: Optional[float] = None
    ELEVATORS_MODE: Optional[float] = None
    ENTRANCES_MODE: Optional[float] = None
    FLOORSMAX_MODE: Optional[float] = None
    FLOORSMIN_MODE: Optional[float] = None
    LANDAREA_MODE: Optional[float] = None
    LIVINGAPARTMENTS_MODE: Optional[float] = None
    LIVINGAREA_MODE: Optional[float] = None
    NONLIVINGAPARTMENTS_MODE: Optional[float] = None
    NONLIVINGAREA_MODE: Optional[float] = None
    APARTMENTS_MEDI: Optional[float] = None
    BASEMENTAREA_MEDI: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = None
    YEARS_BUILD_MEDI: Optional[float] = None
    COMMONAREA_MEDI: Optional[float] = None
    ELEVATORS_MEDI: Optional[float] = None
    ENTRANCES_MEDI: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    FLOORSMIN_MEDI: Optional[float] = None
    LANDAREA_MEDI: Optional[float] = None
    LIVINGAPARTMENTS_MEDI: Optional[float] = None
    LIVINGAREA_MEDI: Optional[float] = None
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = None
    NONLIVINGAREA_MEDI: Optional[float] = None
    TOTALAREA_MODE: Optional[float] = None
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    FLAG_DOCUMENT_2: Optional[float] = None
    FLAG_DOCUMENT_3: Optional[float] = None
    FLAG_DOCUMENT_4: Optional[float] = None
    FLAG_DOCUMENT_5: Optional[float] = None
    FLAG_DOCUMENT_6: Optional[float] = None
    FLAG_DOCUMENT_7: Optional[float] = None
    FLAG_DOCUMENT_8: Optional[float] = None
    FLAG_DOCUMENT_9: Optional[float] = None
    FLAG_DOCUMENT_10: Optional[float] = None
    FLAG_DOCUMENT_11: Optional[float] = None
    FLAG_DOCUMENT_12: Optional[float] = None
    FLAG_DOCUMENT_13: Optional[float] = None
    FLAG_DOCUMENT_14: Optional[float] = None
    FLAG_DOCUMENT_15: Optional[float] = None
    FLAG_DOCUMENT_16: Optional[float] = None
    FLAG_DOCUMENT_17: Optional[float] = None
    FLAG_DOCUMENT_18: Optional[float] = None
    FLAG_DOCUMENT_19: Optional[float] = None
    FLAG_DOCUMENT_20: Optional[float] = None
    FLAG_DOCUMENT_21: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "NAME_TYPE_SUITE": "Unaccompanied",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
                "FONDKAPREMONT_MODE": "reg oper account",
                "HOUSETYPE_MODE": "block of flats",
                "WALLSMATERIAL_MODE": "Panel",
                "EMERGENCYSTATE_MODE": "No",
                "OCCUPATION_TYPE": "Laborers",
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "CNT_CHILDREN": 1,
                "AMT_INCOME_TOTAL": 135000.0,
                "AMT_CREDIT": 450000.0,
                "AMT_ANNUITY": 22500.0,
                "AMT_GOODS_PRICE": 405000.0
            }
        }

# Load the model when the app starts
model = None

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading model")
    model = joblib.load('lgbm_optuna.joblib')
    logger.info("Model loaded successfully")

@app.post("/predict")
async def predict(data: ApplicationData):
    try:
        logger.debug("Received input data: %s", data.dict(exclude_unset=True))
        input_dict = data.dict(exclude_unset=True)
        input_df = pd.DataFrame([input_dict])
        
        logger.debug("Input DataFrame columns: %s", input_df.columns.tolist())
        for col in categorical_features + numeric_features:
            if col not in input_df.columns:
                input_df[col] = None
        
        logger.debug("Transforming data")
        transformed_data = model.named_steps['preprocessor'].transform(input_df)
        logger.debug("Transformed data shape: %s", transformed_data.shape)
        
        logger.debug("Making prediction")
        probability = model.named_steps['classifier'].predict_proba(transformed_data)[:, 1][0]
        
        logger.debug("Prediction: %s", probability)
        return {
            "probability": float(probability),
            "status": "success"
        }
    except Exception as e:
        logger.error("Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Credit Risk Prediction API is running"}