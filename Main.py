from dotenv import load_dotenv
import os
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from Get_Features import get_features
from FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date

load_dotenv("dev.env")
API_KEY = os.getenv("COINGLASS_API_KEY")

if not API_KEY:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")


if __name__ == "__main__":

    # Собираем фичи в один датасет
    getter = FeaturesGetter(api_key=API_KEY)
    
    dfs = get_features(getter, API_KEY)
    
    df_all = merge_by_date(dfs, how="outer", dedupe="last")
    # print(df_all.columns)

    