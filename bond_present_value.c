#include <math.h>
#include <stdio.h>

double PV(float face_value, float coupon_rate, float yield_rate, float maturity_period, float compounding_periods_per_year);

int main()
{
    // get value at maturity
    printf("Face value: ");
    float face_value;
    scanf("%f", &face_value);

    // get market interest rate (not as a %)
    printf("Coupon rate: ");
    float coupon_rate;
    scanf("%f", &coupon_rate);

    // get yield rate (not as a %)
    printf("Yield rate: ");
    float yield_rate;
    scanf("%f", &yield_rate);

    // get time period
    printf("Maturity period: ");
    float maturity_period;
    scanf("%f", &maturity_period);

    // get compounding periods
    printf("Compounding periods per year: ");
    float compounding_periods_per_year;
    scanf("%f", &compounding_periods_per_year);



    printf("%f\n", PV(face_value, coupon_rate, yield_rate, maturity_period, compounding_periods_per_year));
}

double PV(float face_value, float coupon_rate, float yield_rate, float maturity_period, float compounding_periods_per_year)
{
    // Coupon payment per period
    double coupon = face_value * coupon_rate / compounding_periods_per_year;

    // Total number of periods
    int total_periods = maturity_period * compounding_periods_per_year;

    // Initial discounted cash flows
    double dcf = 0;

    // Adjust yield rate for compounding periods
    double adjusted_yield = yield_rate / compounding_periods_per_year;

    // Calculate discounted cash flows
    for (int i = 1; i <= total_periods; i++)
    {
        // Discount factor: (1 + yield rate / m)^i
        double discountf = pow(1 + adjusted_yield, i);
        if (i < total_periods)
        {
            // Update discounted cash flows for each period until maturity
            dcf += coupon / discountf;
        }
        else // For the final period (maturity)
        {
            // Add face value along with the final coupon payment
            dcf += (coupon + face_value) / discountf;
        }
    }

    // Return the present value
    return dcf;
}
