import pandas as pd
from scipy.stats import shapiro,fligner,f_oneway

c_grp_df = pd.read_csv('/Users/arth/Downloads/groups.tsv', delimiter='\t')

c_grp_df.dropna(inplace=True)

with_feature = c_grp_df.loc[c_grp_df['features'] == 1]['mean_mu']

without_feature = c_grp_df.loc[c_grp_df['features'] == 0]['mean_mu']

shapiro_with = shapiro(with_feature)
shapiro_without = shapiro(without_feature)
fk_homogeneity = fligner(with_feature, without_feature)
anova = f_oneway(with_feature, without_feature)

print(f'{shapiro_with.pvalue} {shapiro_without.pvalue} {fk_homogeneity.pvalue} {anova.pvalue}')


#print(f"{c_grp_df[c_grp_df['features'] == 1]['mean_mu'].mean()} {c_grp_df[c_grp_df['features'] == 0]['mean_mu'].mean()}")