import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

df = pd.read_csv('power_data.csv')
# , dtype={'mouse_id': str,
#                                           'time': datetime,
#                                           'stage': str,
#                                           'experimental_day': str,
#                                           'cohort': str,
#                                           'channel': str,
#                                           'frequency': 'float64',
#                                           'power': 'float64'})

a=9

fig, axes = plt.subplots(1, 1, figsize=(11, 18), sharey=True)
# # axes[0].set_axisbelow(True)
# # axes[0].set_yticks(np.arange(0, 1, step=0.1))
# # axes[0].grid(axis='y')
# sns.lineplot(ax=axes[0], data=df, x="frequency", y="power", hue='stage', palette=sns.color_palette("Set2"), errorbar='sd')
# axes[0].set_title('SD')

# # axes[1].set_axisbelow(True)
# # axes[1].set_yticks(np.arange(0, 1, step=0.1))
# # # axes[1].grid(axis='y')
# sns.lineplot(ax=axes[1], data=df, x="frequency", y="power", hue='stage', palette=sns.color_palette("Set2"), errorbar='se')
# axes[1].set_title('SE')

# # axes[2].set_axisbelow(True)
# # axes[2].set_yticks(np.arange(0, 1, step=0.1))
# # axes[2].grid(axis='y')
# sns.lineplot(ax=axes[2], data=df, x="frequency", y="power", hue='stage', palette=sns.color_palette("Set2"), errorbar='pi')
# axes[2].set_title('PI')

# # axes[3].set_axisbelow(True)
# # axes[3].set_yticks(np.arange(0, 1, step=0.1))
# # axes[3].grid(axis='y')
plt.legend([],[], frameon=False)
# sns.lineplot(ax=axes, data=df.loc[(df['stage'] == 'WAKE') & (df['mouse_id'] == 'M1') & (df['experimental_day'] == 'BS')], hue='time', x="frequency", y="power", palette=sns.color_palette("Set2"), estimator=None)#, errorbar='ci') # , hue='stage'
sns.lineplot(ax=axes, data=df, hue='stage', x="frequency", y="power", palette=sns.color_palette("Set2"), estimator=None)#, errorbar='ci') # , hue='stage'
axes[3].set_title('CI')
plt.savefig('plot.png')