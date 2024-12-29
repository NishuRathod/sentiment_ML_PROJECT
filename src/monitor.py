import pandas as pd
from evidently.report import Report
#from evidently.metrics import DataDriftMetric
from evidently.metrics import DatasetDriftMetric

df = pd.read_csv("data/processed/cleaned_tweets.csv")
df1=pd.read_csv("data/processed/cleaned_data.csv")
df2=pd.concat([df,df1],ignore_index=True)
report = Report(metrics=[DatasetDriftMetric()])
report.run(reference_data=df2, current_data=df2)
report.save_html("reports/data_drift_report.html")
