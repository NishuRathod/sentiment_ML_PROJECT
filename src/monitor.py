import pandas as pd
from evidently.report import Report
#from evidently.metrics import DataDriftMetric
from evidently.metrics import DatasetDriftMetric

df = pd.read_csv("data/processed/cleaned_data.csv")
report = Report(metrics=[DatasetDriftMetric()])
report.run(reference_data=df, current_data=df)
report.save_html("reports/data_drift_report.html")
