import pandas as pd
from src.utils import load_object

df=pd.read_csv('src/notebook/data/validation.csv')
print(df.shape)
print(df.columns)

validatiion_df=df.copy()

load_ids=validatiion_df['load_id']

print(load_ids)

cleaner=load_object('artifacts/cleaner.pkl')

validatiion_df_cln=cleaner(validatiion_df)

processor=load_object('artifacts/preprocessor.pkl')

validation_df_processed=processor.transform(validatiion_df_cln)

model_trainer=load_object('artifacts/model.pkl')

predictions=model_trainer.predict(validation_df_processed)

prediction_df = pd.DataFrame({
    "load_id": load_ids,
    "predicted_rate": predictions
})

prediction_df.to_csv(
    "artifacts/validation-predictions.csv",
    index=False
)
