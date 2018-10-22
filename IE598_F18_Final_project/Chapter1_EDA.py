# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sa Yang

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://github.com/zhli3378/IE598_Machine_Learning_in_Fin_Lab/raw/master/IE598_F18_Final_project/MLF_GP1_CreditScore.csv')
cormat = df.corr()
print(cormat)

hm = pd.DataFrame(df.corr())
plt.pcolor(hm)
plt.title("Correlation Matrix")
plt.xlabel("features")
plt.ylabel("features")
plt.show()

df_columns = ['Sales/Revenues', 'Gross Margin', 'EBITDA',
              'EBITDA Margin', 'Net Income Before Extras',
              'Total Debt', 'Net Debt', 'LT Debt',
              'ST Dets', 'Cash', 'Free Cash Flow',
              'Total Debt/EBITDA', 'Net Debt/EBITDA',
              'Total MV', 'Total Debt/MV', 'Net Debt/MV',
              'CFO/Debt', 'CFO', 'Interest Coverage',
              'Total Liquidity', 'Current Liquidity',
              'Current Liabilities', 'EPS Before Extras',
              'PE', 'ROA', 'ROE']

sns.pairplot(df, size=2.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df.iloc[:, 0:26].values.T)
sns.set(font_scale = 1)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size':5},
                 yticklabels=df_columns,
                 xticklabels=df_columns,
                 )
plt.show()