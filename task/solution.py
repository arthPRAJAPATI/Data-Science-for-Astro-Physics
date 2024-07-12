import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, fligner, f_oneway, ks_2samp, pearsonr
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord

# Initialize the cosmology model with specified parameters
my_cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

# Read the datasets
galaxy_coord_df = pd.read_csv('/Users/arth/Downloads/galaxies_coordinates.tsv', delimiter='\t')
c_grp_df = pd.read_csv('/Users/arth/Downloads/groups.tsv', delimiter='\t')

# Drop missing values
c_grp_df.dropna(inplace=True)
galaxy_coord_df.dropna(inplace=True)

# Extract redshifts
z = c_grp_df['z']

# Calculate the angular diameter distances for each group's redshift
c_grp_df['dA'] = c_grp_df['z'].apply(lambda z: my_cosmo.angular_diameter_distance(z).to(u.kpc).value)

# Group the galaxy coordinates by 'Group'
grouped_galaxy = galaxy_coord_df[['Group', 'RA', 'DEC']].groupby(['Group'])

# Initialize dictionaries to store median separations and separations
median_sep = {}

# Loop through each group
for group_name in grouped_galaxy:
    group_data = grouped_galaxy.get_group(group_name[0])
    sep = []
    # Create all possible pairs of galaxies within the group
    combination = itertools.combinations(zip(group_data['RA'], group_data['DEC']), 2)
    dA = c_grp_df.loc[c_grp_df['Group'] == str(group_name[0]).strip("('',)"), 'dA']
    if dA.empty:
        continue
    for (ra1, dec1), (ra2, dec2) in combination:
        # Create SkyCoord objects for the galaxies
        p1 = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree, frame="fk5")
        p2 = SkyCoord(ra=ra2 * u.degree, dec=dec2 * u.degree, frame="fk5")
        # Calculate the angular separation
        sep.append(p1.separation(p2).to(u.rad) * dA)
        if sep:
            median_sep.update({str(group_name[0]).strip("(''),"): np.median(sep)})

# Convert the dictionary to a DataFrame
distance_df = pd.DataFrame.from_dict(median_sep, orient='index')
distance_df.index.name = 'Group'
distance_df.columns = ['mean_sep']

# Calculate the mean surface brightness
mean_mu = c_grp_df.groupby('Group').agg({'mean_mu': 'mean'})

# Merge the DataFrame with mean separations and mean surface brightness
mean_mu = mean_mu.merge(distance_df, on=['Group'])
mean_mu.dropna(inplace=True)

# Plot the scatterplot for mean surface brightness and projected median separation
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

# Conduct the Shapiro-Wilk test for normality on the projected median separation and mean surface brightness
shapiro_sep = shapiro(mean_mu['mean_sep'])
shapiro_mu = shapiro(mean_mu['mean_mu'])

# Calculate the Pearson correlation coefficient and p-value
pearson_corr, pearson_p = pearsonr(mean_mu['mean_sep'], mean_mu['mean_mu'])

# Extract the projected median separation for the HCG 2 group
hcg2_median_separation = mean_mu.loc[mean_mu.index == 'HCG 2', 'mean_sep'].values[0]

# Print the results
print(f"{hcg2_median_separation:.4f} {shapiro_sep.pvalue:.4f} {shapiro_mu.pvalue:.4f} {pearson_p:.4f}")

# Additional statistical tests and plotting (commented out)
# kolmogorov_smirnov_test = ks_2samp(galaxy_df['n'], isolated_galaxy_df['n'], alternative='two-sided')
# print(f"{kolmogorov_smirnov_test.pvalue}")

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
