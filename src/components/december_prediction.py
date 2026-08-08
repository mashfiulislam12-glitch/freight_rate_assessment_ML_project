import os
import sys
import pandas as pd

from src.utils import load_object
from src.exception import CustomException

def predict_december():
    try:
    
        december_df=pd.read_csv('src/notebook/data/december-chart-inputs.csv')
        print(december_df.shape)
        train_df = pd.read_csv('src/notebook/data/train-test.csv')


        route_df = train_df[
            (train_df["pickup"] == "Lexington") &
            (train_df["delivery"] == "Fort Wayne")
            ].copy()

        print(route_df)

        pickup_lat = route_df["pickup_lat"].median()
        pickup_lon = route_df["pickup_lon"].median()

        delivery_lat = route_df["delivery_lat"].median()
        delivery_lon = route_df["delivery_lon"].median()

        market_index = route_df["market_index"].median()
        quote_signal = route_df["quote_signal"].median()


        prediction_df = december_df.copy()


        prediction_df["pickup_lat"] = pickup_lat

        prediction_df["pickup_lon"] = pickup_lon

        prediction_df["delivery_lat"] = delivery_lat

        prediction_df["delivery_lon"] = delivery_lon

        prediction_df["market_index"] = market_index

        prediction_df["quote_signal"] = quote_signal


        cleaner=load_object('artifacts/cleaner.pkl')

        
        prediction_df_cln=cleaner(prediction_df)

        processor=load_object('artifacts/preprocessor.pkl')
        processed_df=processor.transform(prediction_df_cln)

        trainer=load_object('artifacts/model.pkl')
        predictions=trainer.predict(processed_df)


        december_output = december_df.copy()
        december_output["predicted_rate"] = (
            predictions
        )

        
        output_path = os.path.join("artifacts","december-predictions.csv")
        december_output.to_csv(output_path,index=False)


    except Exception as e:
        raise CustomException(e,sys)

if __name__=='__main__':
    predict_december()