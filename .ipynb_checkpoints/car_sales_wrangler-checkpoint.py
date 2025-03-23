import pandas as pd

class CarSalesDataWrangler:
    """
    A class to clean and preprocess car sales data for analysis.
    """

    def __init__(self, filename):
        """
        Initializes the CarSalesDataWrangler with the dataset.

        Parameters:
        -----------
        filename : str
            The path to the CSV file containing the car sales data.
        """
        self.filename = filename
        self.df = pd.read_csv(filename)

    def wrangle(self):
        """
        Cleans and preprocesses the car sales dataset.

        Returns:
        --------
        pd.DataFrame
            A cleaned and preprocessed DataFrame ready for analysis.
        """
        # Drop unwanted columns
        self.df.drop(
            columns=['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size'],
            inplace=True
        )

        # Rename columns for better understanding
        self.df.rename(
            columns={
                "Engine HP": "HP",
                "Engine Cylinders": "Cylinders",
                "Transmission Type": "Transmission",
                "Driven_Wheels": "Drive_Mode",
                "highway MPG": "MPG_H",
                "city mpg": "MPG_C",
                "MSRP": "Price"
            },
            inplace=True
        )

        # Clean price column
        self.df['Price'] = (self.df['Price'].str.replace(',', '')).astype('int')

        # Add car age
        self.df['car_age'] = 2025 - self.df['Year']

        # Convert column names to lowercase
        self.df.columns = [c.lower() for c in self.df.columns]

        # Drop duplicates
        self.df.drop_duplicates(subset=['make', 'model', 'hp', 'price'], inplace=True)

        # Drop null values
        self.df.dropna(axis=0, inplace=True)

        # Outlier removal using IQR
        numeric_df = self.df.select_dtypes(include=['number'])
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (numeric_df >= lower_bound) & (numeric_df <= upper_bound)
        self.df = self.df[mask.all(axis=1)]

        # Reduce model and make categories
        model_list = self.df['model'].value_counts().sort_values(ascending=False).head(19).keys().to_list()
        self.df['model'] = self.df['model'].apply(lambda x: x if x in model_list else 'Others')

        make_list = self.df['make'].value_counts().head(26).keys().to_list()
        self.df['make'] = self.df['make'].apply(lambda x: x if x in make_list else 'Others')

        # Create new features
        self.df['mpg'] = (self.df['mpg_h'] + self.df['mpg_c']) / 2
        self.df['price_segment'] = self.df['price'].apply(
            lambda x: "Budget" if x <= 10_000_000 else "Mid-Range" if x < 30_000_000 else "Premium" if x < 50_000_000 else "Luxury"
        )
        self.df['age_segment'] = self.df['car_age'].apply(
            lambda x: "New" if x <= 9 else "Fairly New" if x <= 13 else "Used" if x <= 20 else "Old"
        )
        self.df['hp_segment'] = self.df['hp'].apply(
            lambda x: "Economy" if x < 150 else "Standard" if x < 250 else "Performance" if x < 400 else "Sports"
        )
        self.df['fuel_segment'] = self.df['mpg'].apply(
            lambda x: "Gas guzzler" if x < 20 else "Average" if x <= 25 else "Efficient"
        )

        return self.df