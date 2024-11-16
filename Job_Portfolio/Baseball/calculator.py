from scipy.stats import norm
phi_value_upper = norm.cdf(1)
phi_value_lower = norm.cdf(-.98)

print(phi_value_upper)
print(phi_value_lower)
print(phi_value_upper-phi_value_lower)


from scipy.integrate import quad

# Parameters
mu = 50  # Mean of the normal distribution
sigma = 50  # Standard deviation of the normal distribution
lower_bound = 1
upper_bound = 100

# Define the PDF function
def pdf(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

# Define the integrand function
def integrand(x, mu, sigma):
    return x * pdf(x, mu, sigma)

# Perform numerical integration
integral, _ = quad(integrand, lower_bound, upper_bound, args=(mu, sigma))

# Calculate the probability of the range
probability_range = norm.cdf(upper_bound, mu, sigma) - norm.cdf(lower_bound, mu, sigma)

# Calculate the conditional expectation
expected_value = integral / probability_range

print("Expected value E[X | 1  X  100]:", expected_value)


lc_phi_val = norm.pdf(1)
prob_above_100 = 1- norm.cdf(1)
print(lc_phi_val)
print(prob_above_100)
print("Expected Value E[X+20|X>100]: ",50*lc_phi_val/prob_above_100+70)