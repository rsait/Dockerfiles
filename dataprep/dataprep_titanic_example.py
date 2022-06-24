from dataprep.datasets import load_dataset
from dataprep.eda import create_report

df = load_dataset("titanic")
report_titanic = create_report(df, title='Report Titanic')
report_titanic.save("/tmp/report_titanic.html")
