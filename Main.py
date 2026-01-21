from dotenv import load_dotenv
import os
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from get_features_from_API import get_features
from FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from CorrelationsAnalyzer.CorrelationsAnalyzer import CorrelationsAnalyzer

load_dotenv("dev.env")
API_KEY = os.getenv("COINGLASS_API_KEY")

if not API_KEY:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")


if __name__ == "__main__":

    getter = FeaturesGetter(api_key=API_KEY)
    engineer = FeaturesEngineer()
    analyzer = CorrelationsAnalyzer()

    # Собираем фичи в один датасет    
    dfs = get_features(getter, API_KEY)
    
    df_all = merge_by_date(dfs, how="outer", dedupe="last")
    # print(df_all.columns)

    # Нормализация спот-колонок
    df0 = engineer.ensure_spot_prefix(df_all)

    # Добавляем целевую колонку
    df1 = engineer.add_y_up_1d(df0)

    df2 = engineer.add_engineered_features(df1)

