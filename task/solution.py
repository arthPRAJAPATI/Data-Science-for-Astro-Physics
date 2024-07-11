import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, fligner, f_oneway, ks_2samp, pearsonr
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord

my_cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

# Read the datasets
galaxy_coord_df = pd.read_csv('/Users/arth/Downloads/galaxies_coordinates.tsv', delimiter='\t')

c_grp_df = pd.read_csv('/Users/arth/Downloads/groups.tsv', delimiter='\t')

# Drop missing values
c_grp_df.dropna(inplace=True)

z = c_grp_df['z']
angular_diameter_distance = my_cosmo.angular_diameter_distance(z).to(u.kpc)
# print(angular_diameter_distance)

grouped_galaxy = galaxy_coord_df[['Group','RA','DEC']].groupby(['Group'])
median_sep = {}
sep = []
for group_name in grouped_galaxy:
    group_data = grouped_galaxy.get_group(group_name[0])
    combination = itertools.product(group_data['RA'], group_data['DEC'])
    combination_iterator = iter(combination)
    for ra, dec, in combination_iterator:
        try:
            next_ra, next_dec = next(combination_iterator)
            p1 = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="fk5")
            p2 = SkyCoord(ra=next_ra * u.degree, dec=next_dec * u.degree, frame="fk5")
            sep.append(p1.separation(p2).value)
        except StopIteration:
            median_sep.update({ str(group_name[0]).strip("(''),"): np.median(sep)})
            sep.clear()

distance_df = pd.DataFrame.from_dict(median_sep, orient='index')
distance_df.index.name = 'Group'
distance_df.columns = ['mean_sep']

mean_mu = c_grp_df.groupby('Group').agg({'mean_mu': 'mean'})

mean_mu = mean_mu.merge(distance_df, on=['Group'])

plt.xlabel('R <kpc>')
plt.rcParams.update({
    "text.usetex": True,
})
plt.ylabel(r'$\mu_{|GL,r} \, (\mathrm{mag~arcsec^{-2}})$')
plt.rcParams.update({
    "text.usetex": False,
})
plt.scatter(mean_mu['mean_sep'], mean_mu['mean_mu'])
plt.show()

# Perform the Shapiro-Wilk test for normality on mean_sep, and mean_mu
shapiro_sep = shapiro(mean_mu['mean_sep'])
shapiro_mu = shapiro(mean_mu['mean_mu'])

# Calculate the Pearson correlation coefficients and p-values
pearson_n = pearsonr(mean_mu['mean_sep'], mean_mu['mean_mu'])

print(f"{mean_mu.loc['HCG 2', 'mean_sep']:.4e} {shapiro_sep.pvalue:.4e} {shapiro_mu.pvalue:.4e} {pearson_n.pvalue:.4e}")

# Read the galaxies_morphology dataset
# morphology_galaxy_df = pd.read_csv('/Users/arth/Downloads/galaxies_morphology.tsv', delimiter='\t')

# Calculate the mean Sérsic index (n) and mean T for each group in the morphology_galaxy_df
# mean_sersic_df = morphology_galaxy_df.groupby(['Group']).agg(
#     mean_n=('n', 'mean'),
#     mean_T=('T', 'mean')
# )

# Merge the mean Sérsic index and mean T values with the c_grp_df dataframe based on the 'Group' column
# new_data = c_grp_df.merge(mean_sersic_df, on=['Group'])

# Extract the mean_n, mean_T, and mean_mu columns as numpy arrays
# mean_n = new_data['mean_n'].values
# mean_T = new_data['mean_T'].values
# mean_mu = new_data['mean_mu'].values

# Plot scatter plot of mean_n vs. mean_mu
# plt.xlabel('<n>')
# plt.rcParams.update({
#     "text.usetex": True,
# })
# plt.ylabel(r'$\mu_{|GL,r} \, (\mathrm{mag~arcsec^{-2}})$')
# plt.rcParams.update({
#     "text.usetex": False,
# })
# plt.scatter(mean_n, mean_mu)
# plt.show()

# Plot scatter plot of mean_T vs. mean_mu
# plt.xlabel('<T>')
# plt.rcParams.update({
#     "text.usetex": True,
# })
# plt.ylabel(r'$\mu_{|GL,r} \, (\mathrm{mag~arcsec^{-2}})$')
# plt.rcParams.update({
#     "text.usetex": False,
# })
# plt.scatter(mean_T, mean_mu)
# plt.show()

# Perform the Shapiro-Wilk test for normality on mean_n, mean_T, and mean_mu
# shapiro_n = shapiro(mean_n)
# shapiro_T = shapiro(mean_T)
# shapiro_mu = shapiro(mean_mu)

# Calculate the Pearson correlation coefficients and p-values for mean_n vs. mean_mu and mean_T vs. mean_mu
# pearson_n = pearsonr(mean_n, mean_mu)
# pearson_T = pearsonr(mean_T, mean_mu)

# Print the p-values from the Shapiro-Wilk tests and the Pearson correlation tests
# print(f"{shapiro_mu.pvalue} {shapiro_n.pvalue} {shapiro_T.pvalue} {pearson_n.pvalue} {pearson_T.pvalue}")

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
