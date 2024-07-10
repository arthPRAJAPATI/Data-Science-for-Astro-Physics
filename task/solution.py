import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, fligner, f_oneway, ks_2samp, pearsonr

# Read the datasets
c_grp_df = pd.read_csv('/Users/arth/Downloads/groups.tsv', delimiter='\t')

# Drop missing values
c_grp_df.dropna(inplace=True)

# Read the galaxies_morphology dataset
morphology_galaxy_df = pd.read_csv('/Users/arth/Downloads/galaxies_morphology.tsv', delimiter='\t')

mean_sersic_df = morphology_galaxy_df.groupby(['Group']).agg(mean_n=('n', 'mean'),mean_T=('T', 'mean'))

new_data = c_grp_df.merge(mean_sersic_df, on=['Group'])

mean_n = new_data['mean_n'].values
mean_T = new_data['mean_T'].values
mean_mu = new_data['mean_mu'].values

plt.xlabel('<n>')
plt.rcParams.update({
    "text.usetex": True,
})
plt.ylabel(r'$\mu_{|GL,r} \, (\mathrm{mag~arcsec^{-2}})$')
plt.rcParams.update({
    "text.usetex": False,
})
plt.scatter(mean_n, mean_mu)
plt.show()

plt.xlabel('<T>')
plt.rcParams.update({
    "text.usetex": True,
})
plt.ylabel(r'$\mu_{|GL,r} \, (\mathrm{mag~arcsec^{-2}})$')
plt.rcParams.update({
    "text.usetex": False,
})
plt.scatter(mean_T, mean_mu)
plt.show()

shapiro_n = shapiro(mean_n)
shapiro_T = shapiro(mean_T)
shapiro_mu = shapiro(mean_mu)

pearson_n = pearsonr(mean_n, mean_mu)
pearson_T = pearsonr(mean_T, mean_mu)

print(f"{shapiro_mu.pvalue} {shapiro_n.pvalue} {shapiro_T.pvalue} {pearson_n.pvalue} {pearson_T.pvalue}")
# Read the isolated_galaxies dataset
# isolated_galaxy_df = pd.read_csv('/Users/arth/Downloads/isolated_galaxies.tsv', delimiter='\t')

# Define the bins for the histogram
# bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Define the names for the legend
# names = ['Morphology Galaxy', 'Isolated Galaxy']

# Plot the histogram for the Sérsic index (n) for both datasets
# plt.hist([morphology_galaxy_df['n'], isolated_galaxy_df['n']], bins=bins, label=names)
# plt.title("Sérsic index (n)")
# plt.legend()
# plt.show()

# Calculate the fraction of galaxies with Sérsic index n > 2 for morphology_galaxy_df
# morphology_fraction = len(morphology_galaxy_df[morphology_galaxy_df['n'] > 2]) / len(morphology_galaxy_df['n'])

# Calculate the fraction of galaxies with Sérsic index n > 2 for isolated_galaxy_df
# isolated_fraction = len(isolated_galaxy_df[isolated_galaxy_df['n'] > 2]) / len(isolated_galaxy_df['n'])

# Perform the Kolmogorov-Smirnov test to compare the Sérsic index distributions
# kolmogorov_smimrov_test = ks_2samp(morphology_galaxy_df['n'], isolated_galaxy_df['n'], alternative='two-sided')

# Print the fractions and the p-value from the Kolmogorov-Smirnov test
# print(f"{morphology_fraction} {isolated_fraction} {kolmogorov_smimrov_test.pvalue}")

# Separate the data into two groups based on the 'features' column
# with_feature = c_grp_df.loc[c_grp_df['features'] == 1]['mean_mu']
# without_feature = c_grp_df.loc[c_grp_df['features'] == 0]['mean_mu']

# Perform the Shapiro-Wilk test for normality on both groups
# shapiro_with = shapiro(with_feature)
# shapiro_without = shapiro(without_feature)

# Perform the Fligner-Killeen test for homogeneity of variances
# fk_homogeneity = fligner(with_feature, without_feature)

# Perform one-way ANOVA to compare the means of the two groups
# anova = f_oneway(with_feature, without_feature)

# Print the p-values from the Shapiro-Wilk, Fligner-Killeen, and ANOVA tests
# print(f'{shapiro_with.pvalue} {shapiro_without.pvalue} {fk_homogeneity.pvalue} {anova.pvalue}')

# Uncomment the following line if you need to print the mean values of 'mean_mu' for the two groups
# print(f"{c_grp_df[c_grp_df['features'] == 1]['mean_mu'].mean()} {c_grp_df[c_grp_df['features'] == 0]['mean_mu'].mean()}")
