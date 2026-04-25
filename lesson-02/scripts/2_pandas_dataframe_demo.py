"""
Simple demonstration of pandas DataFrames.

This script shows basic operations with pandas DataFrames including:
- Creating a DataFrame
- Viewing data
- Selecting columns and rows
- Filtering data
- Basic statistics
- Adding/modifying columns
"""

import pandas as pd


def main():
    """Main function demonstrating pandas DataFrame operations."""
    print("=" * 70)
    print("Pandas DataFrame Basic Examples")
    print("=" * 70)
    
    # 1. Creating a DataFrame from a dictionary
    print("\n1. Creating a DataFrame from a dictionary:")
    print("-" * 70)
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'],
        'Salary': [50000, 60000, 70000, 55000, 65000]
    }
    df = pd.DataFrame(data)
    print(df)
    
    # 2. Viewing basic information about the DataFrame
    print("\n2. DataFrame Info:")
    print("-" * 70)
    print(f"Shape (rows, columns): {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    
    # 3. Viewing first and last rows
    print("\n3. First 3 rows:")
    print("-" * 70)
    print(df.head(3))
    
    print("\n4. Last 2 rows:")
    print("-" * 70)
    print(df.tail(2))
    
    # 4. Selecting a single column
    print("\n5. Selecting a single column (Name):")
    print("-" * 70)
    names = df['Name']
    print(names)
    print(f"\nType: {type(names)}")
    
    # 5. Selecting multiple columns
    print("\n6. Selecting multiple columns (Name and Age):")
    print("-" * 70)
    name_age = df[['Name', 'Age']]
    print(name_age)
    
    # 6. Filtering rows based on conditions
    print("\n7. Filtering: People older than 30:")
    print("-" * 70)
    older_than_30 = df[df['Age'] > 30]
    print(older_than_30)
    
    print("\n8. Filtering: People with salary greater than 55000:")
    print("-" * 70)
    high_salary = df[df['Salary'] > 55000]
    print(high_salary)
    
    # 7. Basic statistics
    print("\n9. Basic Statistics:")
    print("-" * 70)
    print(f"Mean Age: {df['Age'].mean():.2f}")
    print(f"Median Age: {df['Age'].median():.2f}")
    print(f"Max Salary: ${df['Salary'].max():,}")
    print(f"Min Salary: ${df['Salary'].min():,}")
    print(f"Average Salary: ${df['Salary'].mean():,.2f}")
    
    print("\n10. Summary statistics for numeric columns:")
    print("-" * 70)
    print(df.describe())
    
    # 8. Adding a new column
    print("\n11. Adding a new column (Bonus = 10% of Salary):")
    print("-" * 70)
    df['Bonus'] = df['Salary'] * 0.10
    print(df)
    
    # 9. Modifying values
    print("\n12. Modifying a value (Alice's age to 26):")
    print("-" * 70)
    df.loc[df['Name'] == 'Alice', 'Age'] = 26
    print(df[df['Name'] == 'Alice'])
    
    # 10. Sorting
    print("\n13. Sorting by Salary (descending):")
    print("-" * 70)
    sorted_df = df.sort_values('Salary', ascending=False)
    print(sorted_df)
    
    # 11. Creating DataFrame from a list of lists
    print("\n14. Creating DataFrame from a list of lists:")
    print("-" * 70)
    data_list = [
        ['Product A', 100, 10.50],
        ['Product B', 200, 15.75],
        ['Product C', 150, 12.00]
    ]
    df2 = pd.DataFrame(data_list, columns=['Product', 'Quantity', 'Price'])
    df2['Total'] = df2['Quantity'] * df2['Price']
    print(df2)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("Key pandas DataFrame operations:")
    print("  - pd.DataFrame() - Create a DataFrame")
    print("  - df.head() / df.tail() - View first/last rows")
    print("  - df['column'] - Select a column")
    print("  - df[['col1', 'col2']] - Select multiple columns")
    print("  - df[df['column'] > value] - Filter rows")
    print("  - df.describe() - Get summary statistics")
    print("  - df.sort_values() - Sort by column")
    print("  - df.loc[] - Access/modify specific values")
    print("=" * 70)


if __name__ == "__main__":
    main()
