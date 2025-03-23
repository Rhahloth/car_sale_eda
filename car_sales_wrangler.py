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
        df = self.df.copy()  # Work on a copy to minimize repeated calls

        # Drop unwanted columns
        df.drop(
            columns=['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size'],
            inplace=True
        )

        # Rename columns for better understanding
        df.rename(
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
        df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(int)

        # Add car age
        df['car_age'] = 2025 - df['Year']

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()

        # Drop duplicates
        df.drop_duplicates(subset=['make', 'model', 'hp', 'price'], inplace=True)

        # Drop null values
        df.dropna(inplace=True)

        # Outlier removal using IQR
        numeric_df = df.select_dtypes(include=['number'])
        q1, q3 = numeric_df.quantile(0.25), numeric_df.quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (numeric_df >= lower_bound) & (numeric_df <= upper_bound)
        df = df[mask.all(axis=1)]

        # Reduce model and make categories
        top_models = df['model'].value_counts().nlargest(19).index
        df['model'] = df['model'].where(df['model'].isin(top_models), 'Others')

        top_makes = df['make'].value_counts().nlargest(26).index
        df['make'] = df['make'].where(df['make'].isin(top_makes), 'Others')

        # Create new features
        df['mpg'] = (df['mpg_h'] + df['mpg_c']) / 2
        df['price_segment'] = pd.cut(df['price'], bins=[0, 10_000_000, 30_000_000, 50_000_000, float('inf')],
                                     labels=["Budget", "Mid-Range", "Premium", "Luxury"], right=False)
        df['age_segment'] = pd.cut(df['car_age'], bins=[0, 9, 13, 20, float('inf')],
                                   labels=["New", "Fairly New", "Used", "Old"], right=False)
        df['hp_segment'] = pd.cut(df['hp'], bins=[0, 150, 250, 400, float('inf')],
                                  labels=["Economy", "Standard", "Performance", "Sports"], right=False)
        df['fuel_segment'] = pd.cut(df['mpg'], bins=[0, 20, 25, float('inf')],
                                    labels=["Gas guzzler", "Average", "Efficient"], right=False)

        self.df = df  # Assign back to self.df once at the end
        return df
