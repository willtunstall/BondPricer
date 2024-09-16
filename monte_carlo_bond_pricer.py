import ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import requests
from datetime import datetime, date, time, timedelta


# Importing present value function from bond_pricer.c
present_value = ctypes.CDLL("/Users/willtunstall/Desktop/Projects/FinanceProjects/bond_pricer/bond_present_value.so").PV
present_value.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
present_value.restype = ctypes.c_double

# Calculations for bond price convexity and macaulay duration are based on Timothy Falcon Crack and Sanjay K. Nawalkha, "Common Misunderstandings Concerning Duration and Convexity" 
# (https://www.sfu.ca/~poitras/dur_con.pdf)

class BondData:
    def __init__(self, apikey, ticker):
        self.apikey = apikey
        self.ticker = ticker

    def hist_yields(self):

        # Get data on 10-year treasuries from FRED API
        response = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={self.ticker}&api_key={self.apikey}&file_type=json")

        # Test API response
        if response.status_code == 200:

            # Convert data to DataFrame 
            data_json = response.json()
            data = pd.DataFrame(pd.Series(data_json)["observations"])

            # Market yield = most recent yield rate
            market_yield = float(data["value"].iloc[-1:].values[0]) / 100
    
            # Most recent recorded yield date
            current_date = str(data["date"].iloc[-1:].values[0])

            # Year before most recent recorded date
            past_year_date = pd.to_datetime(date.fromisoformat(current_date) - timedelta(days=365))

            # Converting DataFrame dtypes to datetime
            dates = pd.to_datetime(data["date"])
    
            # Checking whether data had been recorded year earlier
            index_past_year = dates[dates == past_year_date].index.tolist()

            # If no data recorded, decrease time range by incriments of a day
            while not index_past_year and past_year_date < dates.max():
                past_year_date += timedelta(days=1)
            index_past_year = dates[dates == past_year_date].index.tolist()

            # First value should be the index of the yield rate at previous year
            index_past_year = index_past_year[0]

            # Using the index to slice the data from past year point until present
            hist_yields = np.array(data["value"].iloc[index_past_year:].replace({'.': 0.0}), dtype=float)
    
            return market_yield, hist_yields
        else:
            raise ValueError("Could not fetch historical yield data from API.")
    
    def yield_volatility(self, past_yields):

        # Calculating volatility of past yield rates
        volatility = np.std(past_yields)
        return volatility / 100
    
class MonteCarloBondPricer:
    def __init__(self, face_value: float, coupon_rate: float, maturity_period: float, n_simulations: int, compounding_periods_per_year: int):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_period = maturity_period
        self.n_simulations = n_simulations
        self.compounding_periods_per_year = compounding_periods_per_year

    def simulated_yield_rates(self, market_yield: float, volatility: float):

        # Simulation of random normal distribution, mean = yield rate, standard deviation = volatility, size = number of simulations
        return np.random.normal(market_yield, volatility, self.n_simulations)
    
    def bond_price(self, market_yield: float, volatility: float):

        # Get simulated yield rates
        yield_rates = self.simulated_yield_rates(market_yield, volatility)

        # Create an array of size n_simulations
        bond_prices = np.zeros(self.n_simulations)

        # Calculates present value for all simulations
        for i in range(self.n_simulations):

            # Implementing global present value function from C shared library
            if self.compounding_periods_per_year > 0 and type(self.compounding_periods_per_year) == int:

                # C function performs discrete compounding of cash flows (could be substituted for a continuosly compounding pricing function)
                bond_prices[i] = present_value(self.face_value, self.coupon_rate, yield_rates[i], self.maturity_period, self.compounding_periods_per_year)
            else:
                sys.exit("Compounding Periods must be greater than 0 and int (DEFAULT = 1)")
                
        # Initialzing the arrays for duration, convexity, modified duration and price changes
        macaulay_duration = np.zeros(self.n_simulations)
        convexity = np.zeros(self.n_simulations)
        modified_duration = np.zeros(self.n_simulations)
        price_changes = np.zeros(self.n_simulations)

        # Aggregating discrete calculations (approx. integral approach)
        for i in range(self.n_simulations):

            # Initialize Macaulay duration and convexity for each simulation
            macaulay_duration[i] = 0
            convexity[i] = 0
            
            # Discrete calculation of Macaulay duration and convexity applied period by period
            if self.coupon_rate == 0:

                # For zero-coupon bond
                # macaulay duration = T
                macaulay_duration[i] = self.maturity_period

                # convexity = (T * (T + 1)) / (1 + r)**2
                convexity[i] = self.maturity_period * (self.maturity_period + 1) / (1 + yield_rates[i])**2

            else:

                for t in range(1, self.maturity_period * self.compounding_periods_per_year + 1):
                    # Time period 
                    time_period = t / self.compounding_periods_per_year

                    # Coupon payment 
                    coupon_payment =(self.coupon_rate * self.face_value) / self.compounding_periods_per_year

                    # Discrete calculations of duration and convexity (sum of discretes to approximate the integral)
                    # macaulay duration [i] = ((t / m) * C) / ((1 + r[i]) / m)**t
                    macaulay_duration[i] += (time_period * coupon_payment) / ((1 + yield_rates[i] / self.compounding_periods_per_year)**t)

                    # convexity [i] = ((t / m) * ((t / m) + 1) * C) / ((1 + r) / m)**(t + 2)
                    convexity[i] += (time_period * (time_period + 1) * coupon_payment) / ((1 + yield_rates[i] / self.compounding_periods_per_year)**(t + 2))

                # (final Macaulay Duration): (T * FV) / (1 + r / m)**(T * m)
                macaulay_duration[i] += (self.maturity_period * self.face_value) / (1 + yield_rates[i] / self.compounding_periods_per_year)**(self.maturity_period * self.compounding_periods_per_year)

                # (final Convexity): (t * (t + 1)) * FV / (1 + r)**(T + 2)
                convexity[i] += (self.maturity_period * (self.maturity_period + 1)) * self.face_value / (1 + yield_rates[i])**(self.maturity_period + 2) 

            # Normalize by the bond price to get Macaulay duration and convexity
            if bond_prices[i] > 0:
                macaulay_duration[i] = (1 / bond_prices[i]) * macaulay_duration[i]
                convexity[i] = (1 / bond_prices[i]) * convexity[i]

            # Calculate Modified Duration
            modified_duration[i] = macaulay_duration[i] / (1 + yield_rates[i] / self.compounding_periods_per_year) 

            # Calculate price change using modified duration
            price_changes[i] = -modified_duration[i] * bond_prices[i] * volatility + 0.5 * convexity[i] * bond_prices[i] * (volatility)**2

            # Updated bond price with price changes
            bond_prices[i] += price_changes[i] 
        
        return bond_prices, macaulay_duration, convexity
            
def main():

    # Obtain FRED API key from command line
    bond_data = BondData(sys.argv[1], "DGS10")

    # Market yield rate (can also be mean yield rate)
    market_yield, hist_yields = bond_data.hist_yields()

    # Yield rate volatility
    volatility = bond_data.yield_volatility(hist_yields)

    # Assumptions based on 10-year Treasury bond
    face_value = 100 # Face value of the government bond
    coupon_rate = 0.0432 # Coupon rate of the bond
    maturity_period = 10 # Years until maturity
    num_simulations = 10000 # Number of Monte Carlo simulations
    compounding_periods_per_year = 2 # compounding periods per year (m) !DEFAULT IS 1 NOT 0!

    # Monte Carlo simulation for yield rates
    bond_pricer = MonteCarloBondPricer(face_value, coupon_rate, maturity_period, num_simulations, compounding_periods_per_year)

    # Adjusted price, duration and convexity
    bond_prices, macaulay_duration, convexity = bond_pricer.bond_price(market_yield, volatility)

    # Estimate of bond price with a mean bond yield
    estimate_price = np.array(bond_prices).sum() / num_simulations

    # Standard deviation and variance of prices in array
    standard_deviation, variance = (np.array(bond_prices).std(), 
                                    np.array(bond_prices).var())
    
    # Standard error of the estimated prices
    standard_error = standard_deviation / np.sqrt(num_simulations)
    
    # Present value using mean yield
    pv = present_value(face_value, coupon_rate, market_yield, maturity_period, compounding_periods_per_year)

    # Print out parameters for analysis
    print(f"Estimated price: {estimate_price} \nStandard deviation: {standard_deviation} \nVariance: {variance} \nStandard error: {standard_error} \nPresent value: {pv}")

    # Call plot function
    plot_duration_and_convexity_distribution(macaulay_duration, convexity)

def plot_duration_and_convexity_distribution(macaulay_duration: np.array, convexity: np.array):
    plt.figure(figsize=(14, 6))

    # Sub-plot 1: Macaulay Duration Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(macaulay_duration, kde=True)
    plt.title('Macaulay Duration Distribution')
    plt.xlabel('Macaulay Duration')
    plt.ylabel('Frequency')

    # Sub-plot 2: Convexity Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(convexity, kde=True)
    plt.title('Convexity Distribution')
    plt.xlabel('Convexity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()