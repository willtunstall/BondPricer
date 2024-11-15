# Bond Pricer
**Author:** William Tunstall

This project is a vanilla and zero-coupon bond pricer that takes a set of assumptions and uses Monte Carlo simulations to compute an estimated bond price. I have use Monte Carlo simulations as they are commonly used throughout finance to account for the randomness of variables. In this case, yield rates.

The project is in two parts. The first is a program written in C that calculates the present value of a bond. Second is a Python script that runs the simulations and calculates price adjustments to estimate a value. Both programs are connected using the ctypes module that allows for C functions to be called in the Python script.

I have also used the FRED API to input some real world data on bond yield rates for a 10-Year U.S. Treasury Bond, but this can easily be replaced with other data.

The bond pricing formula used is based on a paper by Timothy Falcon Crack and Sanjay K. Nawalkha: [“Common Misunderstandings Concerning Duration and Convexity”](https://www.sfu.ca/~poitras/dur_con.pdf). The use of Monte Carlo simulation is based on ideas by [Coding Jesus](https://www.youtube.com/@CodingJesus) and a Medium article by Thomas B [“Government bond pricing with Monte Carlo analysis”](https://medium.com/@bundy01/government-bond-pricing-with-monte-carlo-analysis-fe4c03e43a0a). I will now break down the steps I took to create the bond pricing model.

Starting with the present value formula:

<img src="https://latex.codecogs.com/svg.image?P=\sum_{t=1}^T\frac{C_t}{\left(1&plus;\frac{r}{m}\right)^t}&plus;\frac{F}{\left(1&plus;\frac{r}{m}\right)^T}" /><i>     

- <img src="https://latex.codecogs.com/svg.image?\(P\)"/> = present value
- <img src="https://latex.codecogs.com/svg.image?\( F \)"/> = face value of bond
- <img src="https://latex.codecogs.com/svg.image?\( r \)"/> = yield rate
- <img src="https://latex.codecogs.com/svg.image?\( m \)"/> = compounding periods
- <img src="https://latex.codecogs.com/svg.image?\( T \)"/> = total time periods
- <img src="https://latex.codecogs.com/svg.image?\( C_r \)"/> = coupon rate
- <img src="https://latex.codecogs.com/svg.image?\( C_t \)"/> = coupon payment at time <img src="https://latex.codecogs.com/svg.image?\( t \)"/>, where:
  
<img src="https://latex.codecogs.com/svg.image?C_t=F\cdot\frac{C_r}{m}"/><i>

The C function below makes some simplifications such as computing the discount factor to make calculations easier:

<i><img src="https://latex.codecogs.com/svg.image?discount factor = \left( 1 + \frac{r}{m}\right)^t">

```c
double PV(float face_value, float coupon_rate, float yield_rate, float maturity_period, float compounding_periods_per_year)
{
    double coupon = face_value * coupon_rate / compounding_periods_per_year;
    int total_periods = maturity_period * compounding_periods_per_year;
    double dcf = 0;
    double adjusted_yield = yield_rate / compounding_periods_per_year;
    for (int i = 1; i <= total_periods; i++)
    {
        double discountf = pow(1 + adjusted_yield, i);
        if (i < total_periods)
        {
            dcf += coupon / discountf;
        }
        else
        {
            dcf += (coupon + face_value) / discountf;
        }
    }
    return dcf;
}
```


Moving to the Python script before the present value function is called we simulate <img src="https://latex.codecogs.com/svg.image?\ N \"/> yield rates. For each Monte Carlo simulation <img src="https://latex.codecogs.com/svg.image?\ n \"/>, the yield rate has been generated using NumPy's random normal distribution method:

```python
np.random.normal(market_yield, volatility, self.n_simulations)
```
In the code I have calculated an estimated market yield rate and volatility from the FRED API. 

We then iterate through each simulation of yield rates:
```python
yield_rates = self.simulated_yield_rates(market_yield, volatility)
bond_prices = np.zeros(self.n_simulations)

for i in range(self.n_simulations):
            if self.compounding_periods_per_year > 0 and type(self.compounding_periods_per_year) == int:
                bond_prices[i] = present_value(self.face_value, self.coupon_rate, yield_rates[i], self.maturity_period, self.compounding_periods_per_year)
            else:
                sys.exit("Compounding Periods must be greater than 0 and int (DEFAULT = 1)")
```

Using the previous present value formula, we can begin to create a model for pricing adjustments starting with Macaulay Duration. This measures the weighted average time of cash flows until maturity. The formula used for Macaulay Duration is:

<img src="https://latex.codecogs.com/svg.image?D_{Mac}=\frac{1}{P}\sum_{t=1}^T\frac{t\cdot&space;C_t}{\left(1&plus;\frac{r}{m}\right)^t}&plus;\frac{T\cdot&space;F}{\left(1&plus;\frac{r}{m}\right)^T}"/><i>

```python
for i in range(self.n_simulations):
    macaulay_duration[i] = 0
    
    if self.coupon_rate == 0:
        macaulay_duration[i] = self.maturity_period
        
    else:
        for t in range(1, self.maturity_period * self.compounding_periods_per_year + 1):
            time_period_years = t / self.compounding_periods_per_year
            coupon_payment = (self.coupon_rate * self.face_value) / self.compounding_periods_per_year
                    
            macaulay_duration[i] += (time_period_years * coupon_payment) / ((1 + yield_rates[i] / self.compounding_periods_per_year)**t)

        macaulay_duration[i] += (self.maturity_period * self.face_value) / (1 + yield_rates[i] / self.compounding_periods_per_year)**(self.maturity_period * self.compounding_periods_per_year)

    if bond_prices[i] > 0:
        macaulay_duration[i] = (1 / bond_prices[i]) * macaulay_duration[i]
```

In the Python script, this may appear to be inconsistent. This is due to the Macaulay Duration measuring the weighted average in terms of years rather than overall periods. Adjusting for this, we get:

<img src="https://latex.codecogs.com/svg.image?D_{Mac}=\frac{1}{P}\sum_{t=1}^T\frac{\frac{t}{m}\cdot&space;C_t}{\left(1&plus;\frac{r}{m}\right)^t}&plus;\frac{\frac{T}{m}\cdot&space;F}{\left(1&plus;\frac{r}{m}\right)^T}"/><i>


We can then calculate the Convexity of the bond to measure its price sensitivity to yield rate changes. Using the same logic as previously (weights in terms of years), the formula for Convexity is:

<img src="https://latex.codecogs.com/svg.image?C=\frac{1}{P}\sum_{t=1}^T\frac{\frac{t}{m}\left(\frac{t}{m}&plus;1\right)\cdot&space;C_t}{\left(1&plus;\frac{r}{m}\right)^{\left(t&plus;2\right)}}&plus;\frac{\frac{T}{m}\left(\frac{T}{m}&plus;1\right)\cdot&space;F}{\left(1&plus;\frac{r}{m}\right)^{\left(T&plus;2\right)}}"/><i>

```Python
for i in range(self.n_simulations):
    convexity[i] = 0
            
    if self.coupon_rate == 0:
        convexity[i] = self.maturity_period * (self.maturity_period + 1) / (1 + yield_rates[i])**2

    else:
        for t in range(1, self.maturity_period * self.compounding_periods_per_year + 1):
            time_period_years = t / self.compounding_periods_per_year
            coupon_payment = (self.coupon_rate * self.face_value) / self.compounding_periods_per_year

            convexity[i] += (time_period_years * (time_period_years + 1) * coupon_payment) / ((1 + yield_rates[i] / self.compounding_periods_per_year)**(t + 2))
            
        convexity[i] += (self.maturity_period * (self.maturity_period + 1)) * self.face_value / (1 + yield_rates[i])**(self.maturity_period + 2) 

    if bond_prices[i] > 0:
        convexity[i] = (1 / bond_prices[i]) * convexity[i]
```

From Macaulay Duration, we can also calculate the Modified Duration:

<img src="https://latex.codecogs.com/svg.image?D^*=\frac{D_{Mac}}{1&plus;\frac{r}{m}}"/><i>

```python
modified_duration[i] = macaulay_duration[i] / (1 + yield_rates[i] / self.compounding_periods_per_year) 
```

The Modified Duration, like convexity, measures the sensitivity of the bond price to changes in yield rates. Duration is the first-order measure, and convexity is the second-order measure. This topic can be explored more deeply in Crack and Nawalkha's paper, as mentioned previously.

Now we can construct a formula that acts as an approximation for the Taylor Series. Calculating price changes over all the simulations. For this, we use volatility of bond yield rate <img src= "https://latex.codecogs.com/svg.image?\(\Delta&space;r=V\)"/> to represent the change in yield rates:

<img src="https://latex.codecogs.com/svg.image?\Delta&space;P=-D^*\cdot&space;P\cdot&space;V&plus;\frac{1}{2}\cdot&space;P\cdot&space;C\cdot(V)^2"/><i>

```python
price_changes[i] = -modified_duration[i] * bond_prices[i] * volatility + 0.5 * convexity[i] * bond_prices[i] * (volatility)**2
```

We then calculate the expected bond price, factoring in the price changes, over <img src="https://latex.codecogs.com/svg.image?\ N \"/> Monte Carlo simulations:

<img src="https://latex.codecogs.com/svg.image?E(P)=\frac{1}{N}\sum_{n=1}^N&space;P_n-\Delta&space;P_n"/><i>

```python
bond_prices[i] += price_changes[i]
```
```python
estimate_price = np.array(bond_prices).sum() / num_simulations
```

The Python script also outputs graphs that show the Macaulay Duration Distribution and the Convexity Distribution. These can be used to analyze the risk of yield rate changes on a bond's price.
```python I'm A tab
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(macaulay_duration, kde=True)
plt.title('Macaulay Duration Distribution')
plt.xlabel('Macaulay Duration')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(convexity, kde=True)
plt.title('Convexity Distribution')
plt.xlabel('Convexity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```
