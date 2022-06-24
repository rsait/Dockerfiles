from dataprep.eda import create_report
import pandas as pd
import numpy as np

random_data_shape = (1000, 5)
random_data = np.random.uniform(0, 1, random_data_shape)
df_random = pd.DataFrame(random_data, columns = ["Random1", "Random2", "Random3", "Random4", "Random5"])
report_random = create_report(df_random, title='Report Random')
report_random.save("/tmp/report_random.html")
